import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class CreditCardFst(GraphFst):
    """
    Finite state transducer for classifying credit card numbers, e.g.
        4321-8765-2109-6543 -> credit_card { number_part: "四三二一の八七六五の二一零九の六五四三" }
        4321876521096543 -> credit_card { number_part: "四三二一の八七六五の二一零九の六五四三" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="credit_card", kind="classify", deterministic=deterministic)
        
        # TTS-friendly Japanese digit mapping
        tts_digit_map = {
            "0": "ゼロ",
            "1": "イチ", 
            "2": "ニー",
            "3": "サン",
            "4": "ヨン",
            "5": "ゴ",
            "6": "ロク",
            "7": "ナナ",
            "8": "ハチ",
            "9": "キュー"
        }
        
        # Convert each digit
        digit = pynini.union(*[pynini.cross(d, reading) for d, reading in tts_digit_map.items()])
        
        # Define separator
        separator = pynutil.delete("-") | pynutil.delete(" ")
        segment_separator = pynutil.insert("の")
        
        # Credit card number format (16 digits with optional separators)
        # Format: XXXX-XXXX-XXXX-XXXX or XXXXXXXXXXXXXXXX
        number_part = (
            digit + digit + digit + digit +  # First group
            separator + segment_separator +
            digit + digit + digit + digit +  # Second group
            separator + segment_separator +
            digit + digit + digit + digit +  # Third group
            separator + segment_separator +
            digit + digit + digit + digit    # Fourth group
        )
        
        # Final graph with the number part
        final_graph = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize() 