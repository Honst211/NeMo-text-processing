# nemo_text_processing/text_normalization/ja/verbalizers/money.py

import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, delete_space, NEMO_NOT_QUOTE

class MoneyFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # process_negative and process_decimal remain the same
        process_negative = pynini.closure(
            pynutil.delete("negative: \"true\"") + delete_space + pynutil.insert("マイナス"), 0, 1
        )
        process_decimal = pynini.closure(
            delete_space
            + pynutil.delete("decimal_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynutil.insert("銭"),
            0, 1
        )

        # --- Reordering Logic for Currency and Integer --- (保持不变)
        integer_value_content = pynini.closure(NEMO_NOT_QUOTE, 1).optimize()
        yen_reorder_path = (
            pynutil.delete("currency: \"円\"")
            + delete_space
            + pynutil.delete("integer_part: \"")
            + integer_value_content
            + pynutil.delete("\"")
            + pynutil.insert("円")
        )
        non_yen_currency_value = pynini.difference(pynini.closure(NEMO_NOT_QUOTE, 1), "円").optimize()
        extract_non_yen_currency_tag_and_value = (
            pynutil.delete("currency: \"") + non_yen_currency_value + pynutil.delete("\"")
        )
        extract_integer_tag_and_value = (
            pynutil.delete("integer_part: \"") + integer_value_content + pynutil.delete("\"")
        )
        general_ci_path_for_non_yen = (
            extract_non_yen_currency_tag_and_value + delete_space + extract_integer_tag_and_value
        )
        integer_first_path = (
            extract_integer_tag_and_value
            + delete_space
            + pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        currency_integer_logic = (
            yen_reorder_path
            | general_ci_path_for_non_yen
            | integer_first_path
        ).optimize()

        # --- NEW: Optional Suffix "分" Handling ---
        # This will consume the "unit: '分'" tag and insert "分" at the end.
        process_unit_fun = pynini.closure(
            delete_space
            + pynutil.delete("unit: \"分\"") # Consume the tag
            + pynutil.insert("分"),        # Insert the character
            0, 1
        )

        # --- Final Graph Construction --- (MODIFIED)
        graph = (
            process_negative
            + currency_integer_logic
            + process_decimal
            + process_unit_fun # Add the handler for the optional "分" unit
        )

        self.fst = self.delete_tokens(graph).optimize()