import re

class SecurityEngine:
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"system prompt",
        r"reveal your (instructions|system|rules)",
        r"you are now an? (unrestricted|evil|dan)",
        r"disregard context",
        r"print the text above",
        r"execute the following command"
    ]

    @classmethod
    def sanitize_retrieved_text(cls, text: str) -> str:
        cleaned_text = text
        for pattern in cls.INJECTION_PATTERNS:
            cleaned_text = re.sub(pattern, "[MALICIOUS INSTRUCTION NEUTRALIZED]", cleaned_text, flags=re.IGNORECASE)
        return cleaned_text

    @classmethod
    def wrap_context_block(cls, passages: list[dict]) -> str:
        formatted_blocks = []
        for idx, item in enumerate(passages):
            clean_content = cls.sanitize_retrieved_text(item["content"])
            formatted_blocks.append(
                f'<evidence index="{idx + 1}" source="{item["source"]}" page="{item["page"]}" chunk="{item["chunk_id"]}">\n'
                f'{clean_content}\n'
                f'</evidence>'
            )
        return "\n".join(formatted_blocks)
