from app.models.schemas import RetrievedNode, RiskWarning


class WarningsGenerator:
    """Generates risk warnings for medical RAG responses."""

    def __init__(self, hallucination_threshold: float = 0.5):
        self.hallucination_threshold = hallucination_threshold
        self.medication_keywords = [
            "药物",
            "用药",
            "剂量",
            "服药",
            "吃药",
            "药品",
            "药",
        ]
        self.diagnosis_keywords = [
            "诊断",
            "确诊",
            "治疗方案",
        ]
        self.emergency_keywords = [
            "紧急",
            "急诊",
            "立即",
            "马上",
        ]
        self.general_warning = "本回答由AI生成，仅供参考，不能替代专业医疗建议。"

    def generate(
        self,
        answer: str,
        contexts: list[RetrievedNode] | None = None,
        citations: list | None = None,
    ) -> list[RiskWarning]:
        """Generate risk warnings based on answer content and citations."""
        warnings: list[RiskWarning] = []

        warnings.append(
            RiskWarning(
                type="general",
                message=self.general_warning,
                priority="low",
            )
        )

        if self._contains_medication(answer):
            warnings.append(
                RiskWarning(
                    type="medication",
                    message="涉及药物信息，请务必在医生或药师指导下使用。",
                    priority="medium",
                )
            )

        if self._contains_diagnosis(answer):
            warnings.append(
                RiskWarning(
                    type="diagnosis",
                    message="AI无法提供正式医学诊断，请咨询医疗专业人员。",
                    priority="high",
                )
            )

        if self._contains_emergency(answer):
            warnings.append(
                RiskWarning(
                    type="emergency",
                    message="如有紧急症状，请立即就医或拨打急救电话。",
                    priority="high",
                )
            )

        if citations:
            warnings.extend(self._check_hallucination(citations))

        return warnings

    def _contains_medication(self, answer: str) -> bool:
        return any(kw in answer for kw in self.medication_keywords)

    def _contains_diagnosis(self, answer: str) -> bool:
        return any(kw in answer for kw in self.diagnosis_keywords)

    def _contains_emergency(self, answer: str) -> bool:
        return any(kw in answer for kw in self.emergency_keywords)

    def _check_hallucination(self, citations: list) -> list[RiskWarning]:
        """Check for unverified citations indicating hallucination."""
        warnings = []
        unverified = [c for c in citations if not getattr(c, "verified", True)]
        total = len(citations)
        if total > 0 and len(unverified) / total > self.hallucination_threshold:
            warnings.append(
                RiskWarning(
                    type="hallucination",
                    message=f"检测到 {len(unverified)}/{total} 条引用来源无法验证，AI可能存在幻觉。",
                    priority="high",
                )
            )
        return warnings
