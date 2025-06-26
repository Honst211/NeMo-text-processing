import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.inverse_text_normalization.ja.utils import get_abs_path


class CreditCardFst(GraphFst):
    """
    Finite state transducer for classifying credit card numbers
        e.g. 四三二一の八七六五の二一零九の六五四三 -> credit_card { number_part: "4321-8765-2109-6543" }
    """

    def __init__(self):
        super().__init__(name="credit_card", kind="classify")
        
        # TTS-friendly Japanese digit mapping
        tts_digit_map = {
            "ゼロ": "0",
            "イチ": "1", 
            "ニー": "2",
            "サン": "3",
            "ヨン": "4",
            "ゴ": "5",
            "ロク": "6",
            "ナナ": "7",
            "ハチ": "8",
            "キュー": "9"
        }
        
        # Convert each digit
        digit = pynini.union(*[pynini.cross(reading, d) for reading, d in tts_digit_map.items()])
        
        # Define separator
        separator = pynutil.delete("の") + pynutil.insert("-")
        
        # Credit card number format
        number_part = (
            digit + digit + digit + digit +  # First group
            separator +
            digit + digit + digit + digit +  # Second group
            separator +
            digit + digit + digit + digit +  # Third group
            separator +
            digit + digit + digit + digit    # Fourth group
        )
        
        # Final graph with the number part
        final_graph = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize() 