"""
ANALYSIS AGENT
Reads a paper and pulls out the parts a researcher actually cares about, as
clean, structured sections:

  - objectives      : the research objectives / goals
  - methodology     : methods, algorithms, and techniques used
  - datasets_tools  : datasets, tools, and the experimental setup
  - evaluation      : evaluation metrics and performance results
  - limitations     : weaknesses / open problems
  - future_work     : future work suggested by the authors

It returns a dict of these sections (markdown strings) so the UI can show them
and later features (comparison, gap analysis, reports) can reuse them.
"""
import json

from agents.base_agent import BaseAgent

# The sections we extract, with a short label for the UI.
SECTIONS = {
    "objectives": "🎯 Research Objectives",
    "methodology": "🧪 Methodology, Algorithms & Techniques",
    "datasets_tools": "🗄️ Datasets, Tools & Experimental Setup",
    "evaluation": "📊 Evaluation Metrics & Results",
    "limitations": "⚠️ Limitations",
    "future_work": "🔮 Future Work",
}


class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AnalysisAgent",
            system_prompt=(
                "You are a meticulous research analyst. Read the paper and extract "
                "the following, grounded ONLY in what the paper says. Return ONLY a "
                "JSON object with EXACTLY these string keys:\n"
                '  "objectives": the research objectives / goals,\n'
                '  "methodology": methods, algorithms, and techniques used,\n'
                '  "datasets_tools": datasets, tools, and experimental setup,\n'
                '  "evaluation": evaluation metrics and performance results,\n'
                '  "limitations": limitations / weaknesses / open problems,\n'
                '  "future_work": future work suggested by the authors.\n'
                "Each value is a concise markdown string - use short bullet points "
                "('- ') where helpful. If the paper does not mention something, set "
                'that value to "Not reported in the paper." Output JSON only - no '
                "markdown fences, no commentary."
            ),
        )

    @staticmethod
    def _parse_json(raw):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        return json.loads(raw)

    @staticmethod
    def _excerpt(text, head=9000, tail=7000):
        """
        Give the model the START and the END of the paper. Objectives/methods sit
        at the front, while evaluation, limitations, future work, and conclusions
        sit at the very end - a simple head-only slice misses all of those.
        """
        if len(text) <= head + tail:
            return text
        return text[:head] + "\n\n[... middle of paper omitted ...]\n\n" + text[-tail:]

    def analyze(self, text):
        """Return a dict with one entry per section in SECTIONS."""
        result = {key: "Not reported in the paper." for key in SECTIONS}
        try:
            raw = self.run(
                f"Analyze this research paper:\n\n{self._excerpt(text)}",
                temperature=0.2,
            )
            parsed = self._parse_json(raw)
            for key in SECTIONS:
                value = parsed.get(key)
                if value:
                    # The model may return a list; join it into markdown bullets.
                    if isinstance(value, list):
                        value = "\n".join(f"- {v}" for v in value)
                    result[key] = str(value).strip()
        except Exception:
            # Leave the "Not reported" skeleton if the call/parse fails.
            pass
        return result
