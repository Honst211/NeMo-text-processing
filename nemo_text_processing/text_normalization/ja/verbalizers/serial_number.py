from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, NEMO_NOT_QUOTE
import pynini
from pynini.lib import pynutil

class SerialNumberFst(GraphFst):
    def __init__(self, deterministic=True):
        super().__init__(name="serial_number", kind="verbalize", deterministic=deterministic)

        WHITESPACE = pynini.closure(pynini.union(" ", "\n", "\t"))

        prefix = (
            pynutil.delete('prefix: "') +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete('"') +
            pynutil.insert("、")
        )

        number = (
            pynutil.delete('number_part: "') +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete('"')
        )

        graph_full = pynini.closure(prefix + WHITESPACE, 0, 1) + number

        # 支持 value 字段模式
        value = (
            pynutil.delete('value: "') +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete('"')
        )

        graph = graph_full | value

        self.fst = self.delete_tokens(graph).optimize()