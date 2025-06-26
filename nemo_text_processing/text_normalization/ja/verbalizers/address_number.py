import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class AddressNumberFst(GraphFst):
    """
    Finite state transducer for verbalizing address numbers, e.g.
        address_number { number_part: "五七零の八" } -> 五七零の八
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="address_number", kind="verbalize", deterministic=deterministic)
        
        number_part = (
            pynutil.delete("number_part: \"") +
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete("\"")
        )
        
        graph = self.delete_tokens(number_part)
        self.fst = graph 