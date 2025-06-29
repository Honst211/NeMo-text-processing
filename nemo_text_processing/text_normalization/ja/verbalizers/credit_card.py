import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class CreditCardFst(GraphFst):
    """
    Finite state transducer for verbalizing credit card numbers, e.g.
        credit_card { number_part: "四三二一の八七六五の二一零九の六五四三" } -> 四三二一の八七六五の二一零九の六五四三
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="credit_card", kind="verbalize", deterministic=deterministic)
        
        number_part = (
            pynutil.delete("number_part: \"") +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete("\"")
        )
        
        graph = self.delete_tokens(number_part)
        self.fst = graph.optimize() 