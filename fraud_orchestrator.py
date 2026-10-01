"""
FraudX — standalone fraud orchestrator.
This replacement removes the missing core/ and agents/ package dependency
while keeping the API contract expected by app.py.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AgentResult:
    agent_name: str
    fraud_flag: bool
    fraud_probability: float
    confidence: float
    latency_ms: float
    signals: List[str]


@dataclass
class RoutingResult:
    primary_agent: str
    secondary_agents: List[str]
    reason: str


@dataclass
class Decision:
    transaction_id: str
    risk_score: float
    action: str
    primary_agent: str
    agent_scores: Dict[str, float]
    agent_flags: Dict[str, bool]
    risk_signals: List[str]
    recommended_action: str
    confidence: float


class FraudOrchestrator:
    def __init__(self):
        print("🚀 Initialising Fraud Orchestrator...")
        self.atm_model = None
        self.bank_model = None
        self.credit_model = None
        self.graph_accounts = set()
        self.graph_merchants = set()
        self._trained = False
        self._audit_log = []

    @staticmethod
    def _num(series, default=0.0):
        return pd.to_numeric(series, errors="coerce").fillna(default)

    def _fit_model(self, df, cols):
        available = [c for c in cols if c in df.columns]
        if not available or len(df) < 20:
            return None
        X = pd.DataFrame({c: self._num(df[c]) for c in available}).replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(
            n_estimators=120,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )
        model.fit(Xs)
        return (model, scaler, available)

    def train_all(self, atm_path: str, bank_path: str):
        print("\n📚 Training fraud-detection agents...")
        atm_df = pd.read_csv(atm_path)
        bank_df = pd.read_csv(bank_path)

        self.atm_model = self._fit_model(
            atm_df,
            ["No_Of_Withdrawals", "total_amount_withdrawn", "amount_withdrawn_other_card"]
        )

        self.bank_model = self._fit_model(
            bank_df,
            ["TransactionAmount", "CustomerAge", "TransactionDuration",
             "LoginAttempts", "AccountBalance"]
        )

        self.credit_model = self.bank_model

        if "AccountID" in bank_df.columns:
            self.graph_accounts = set(bank_df["AccountID"].astype(str).dropna())
        if "MerchantID" in bank_df.columns:
            self.graph_merchants = set(bank_df["MerchantID"].astype(str).dropna())

        self._trained = True
        print(f"   ATM records: {len(atm_df):,}")
        print(f"   Bank records: {len(bank_df):,}")
        print("\n✅ All agents trained and ready.\n")

    @staticmethod
    def _value(txn, *names, default=None):
        for name in names:
            if name in txn and txn[name] is not None:
                return txn[name]
        return default

    def _normalize(self, raw):
        amount = float(self._value(raw, "amount", "TransactionAmount", default=0) or 0)
        channel = str(self._value(raw, "channel", "Channel", default="online") or "online").lower()
        transaction_type = str(self._value(raw, "transaction_type", "TransactionType", default="Debit") or "Debit")
        account = str(self._value(raw, "account_id", "AccountID", default="UNKNOWN") or "UNKNOWN")
        merchant = str(self._value(raw, "merchant_id", "MerchantID", default="UNKNOWN") or "UNKNOWN")
        location = str(self._value(raw, "location", "Location", default="Unknown") or "Unknown")
        login = float(self._value(raw, "login_attempts", "LoginAttempts", default=1) or 1)
        balance = float(self._value(raw, "balance", "AccountBalance", default=0) or 0)
        duration = float(self._value(raw, "duration_seconds", "TransactionDuration", default=60) or 60)
        age = float(self._value(raw, "customer_age", "CustomerAge", default=35) or 35)
        txn_id = str(self._value(raw, "transaction_id", "TransactionID", default=f"TXN{int(time.time()*1000)}"))
        date = str(self._value(raw, "TransactionDate", default=""))
        try:
            hour = pd.to_datetime(date).hour
        except Exception:
            hour = 12
        return {
            "transaction_id": txn_id, "account_id": account, "merchant_id": merchant,
            "amount": amount, "channel": channel, "transaction_type": transaction_type,
            "location": location, "login_attempts": login, "balance": balance,
            "duration_seconds": duration, "customer_age": age, "hour": hour,
            "is_weekend": hour in (0, 1, 2, 3, 4, 5),
            "is_holiday": False
        }

    def _model_score(self, model_pack, values):
        if not model_pack:
            return 0.0
        model, scaler, cols = model_pack
        row = pd.DataFrame([{c: float(values.get(c, 0)) for c in cols}])
        x = scaler.transform(row)
        raw = float(model.decision_function(x)[0])
        # Map IsolationForest decision score to a simple 0..1 anomaly score.
        return float(np.clip(0.5 - raw, 0, 1))

    def _agent_results(self, txn):
        amount = txn["amount"]
        bank_values = {
            "TransactionAmount": amount,
            "CustomerAge": txn["customer_age"],
            "TransactionDuration": txn["duration_seconds"],
            "LoginAttempts": txn["login_attempts"],
            "AccountBalance": txn["balance"],
        }

        bank_score = self._model_score(self.bank_model, bank_values)
        atm_score = self._model_score(self.atm_model, {
            "No_Of_Withdrawals": 1,
            "total_amount_withdrawn": amount,
            "amount_withdrawn_other_card": amount if txn["channel"] == "atm" else 0,
        })

        # Rule-based signals supplement the anomaly model.
        signals = []
        rule_score = 0.0
        if amount > 1500:
            rule_score += 0.25
            signals.append("high_amount")
        if txn["login_attempts"] >= 3:
            rule_score += 0.20
            signals.append("multiple_login_attempts")
        if txn["balance"] > 0 and amount > txn["balance"]:
            rule_score += 0.25
            signals.append("amount_exceeds_balance")
        if txn["hour"] < 6:
            rule_score += 0.15
            signals.append("unusual_hour")
        if txn["location"].lower() in ("unknown", "suspicious"):
            rule_score += 0.15
            signals.append("unusual_location")

        credit_score = min(1.0, 0.65 * bank_score + 0.35 * rule_score)
        graph_score = 0.0
        if txn["account_id"] not in self.graph_accounts:
            graph_score += 0.25
            signals.append("new_account")
        if txn["merchant_id"] not in self.graph_merchants:
            graph_score += 0.15
            signals.append("new_merchant")
        graph_score = min(1.0, graph_score + 0.35 * rule_score)

        scores = {
            "ATM_AGENT": min(1.0, 0.75 * atm_score + 0.25 * rule_score),
            "BANK_AGENT": min(1.0, 0.70 * bank_score + 0.30 * rule_score),
            "CREDIT_AGENT": credit_score,
            "GRAPH_AGENT": graph_score,
        }
        return scores, signals

    def process(self, raw_transaction: Dict, verbose: bool = True) -> Dict:
        if not self._trained:
            raise RuntimeError("Models are not trained yet.")

        t0 = time.time()
        txn = self._normalize(raw_transaction)
        channel = txn["channel"]

        if channel == "atm":
            primary = "ATM_AGENT"
            secondary = ["BANK_AGENT", "GRAPH_AGENT"]
            reason = "ATM channel routed to ATM and supporting agents"
        elif channel in ("online", "upi", "transfer", "mobile", "pos"):
            primary = "BANK_AGENT"
            secondary = ["CREDIT_AGENT", "GRAPH_AGENT"]
            reason = "Digital transaction routed to bank, credit and graph agents"
        else:
            primary = "BANK_AGENT"
            secondary = ["GRAPH_AGENT"]
            reason = "General transaction routed to bank and graph agents"

        routing = RoutingResult(primary, secondary, reason)
        scores, signals = self._agent_results(txn)
        agent_names = [primary] + secondary
        used_scores = [scores[a] for a in agent_names]
        risk = float(np.clip(0.45 * max(used_scores) + 0.55 * np.mean(used_scores), 0, 1))

        flags = {a: scores[a] >= 0.55 for a in scores}
        action = "BLOCK" if risk >= 0.75 else "ALERT" if risk >= 0.45 else "ALLOW"
        recommended = {
            "BLOCK": "Block transaction and investigate",
            "ALERT": "Allow with additional verification",
            "ALLOW": "Allow transaction"
        }[action]
        confidence = float(np.clip(0.5 + abs(risk - 0.5), 0, 1))

        decision = Decision(
            txn["transaction_id"], round(risk, 4), action, primary,
            {k: round(v, 4) for k, v in scores.items()},
            flags, sorted(set(signals)), recommended, round(confidence, 4)
        )

        latency = round((time.time() - t0) * 1000, 1)
        self._audit_log.append({
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "channel": txn["channel"],
            "risk_score": decision.risk_score,
            "action": decision.action,
            "agents_run": agent_names,
            "agent_scores": decision.agent_scores,
            "signals": decision.risk_signals,
            "total_ms": latency,
        })

        results = [
            AgentResult(a, flags[a], scores[a], confidence, latency / max(len(agent_names), 1), signals)
            for a in agent_names
        ]
        return {"transaction": txn, "routing": routing, "results": results,
                "decision": decision, "latency_ms": latency}

    def process_batch(self, transactions: List[Dict], verbose: bool = False) -> pd.DataFrame:
        rows = []
        for i, txn in enumerate(transactions):
            out = self.process(txn, verbose=verbose)
            d = out["decision"]
            rows.append({
                "idx": i, "transaction_id": d.transaction_id,
                "amount": out["transaction"]["amount"],
                "channel": out["transaction"]["channel"],
                "risk_score": d.risk_score, "action": d.action,
                "primary_agent": d.primary_agent,
                "signals": "|".join(d.risk_signals),
                "latency_ms": out["latency_ms"],
            })
        return pd.DataFrame(rows)

    def get_audit_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._audit_log)
