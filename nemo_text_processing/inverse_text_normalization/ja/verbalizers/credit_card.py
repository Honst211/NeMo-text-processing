import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class CreditCardFst(GraphFst):
    """
    Finite state transducer for verbalizing credit card numbers, e.g.
        credit_card { number_part: "4321-8765-2109-6543" } -> 4321-8765-2109-6543
    """

    def __init__(self):
        super().__init__(name="credit_card", kind="verbalize")
        
        number_part = (
            pynutil.delete("number_part: \"") +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete("\"")
        )
        
        graph = self.delete_tokens(number_part)
 