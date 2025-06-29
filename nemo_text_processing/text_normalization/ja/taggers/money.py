# nemo_text_processing/text_normalization/ja/taggers/money.py

import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, NEMO_DIGIT

class MoneyFst(GraphFst):
    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_numbers = cardinal.just_cardinals

        # --- Currency Definitions --- (保持不变)
        currency_symbols_map = {
            "¥": "円", "$": "ドル", "€": "ユーロ", "£": "ポンド",
            "￥": "円", "＄": "ドル",
        }
        currency_symbols_fst = pynini.string_map(currency_symbols_map.items())
        currency_units_text = ["円", "ドル", "ユーロ", "ポンド"]
        currency_units_fst = pynini.string_map([(x, x) for x in currency_units_text])
        any_currency_form = (currency_symbols_fst | currency_units_fst).optimize()
        currency_component = (
            pynutil.insert("currency: \"") +
            any_currency_form +
            pynutil.insert("\"")
        ).optimize()

        # --- Negative Sign Handling --- (保持不变)
        negative_sign_chars = pynini.union("−", "-").optimize()
        pre_negative_boundary_chars = pynini.union(
            ":", "：", "（", "(", "）", ")", "「", "『", "“", "‘", "、", " "
        ).optimize()
        optional_prefix_boundary = pynini.closure(pre_negative_boundary_chars).optimize()
        process_negative_sign = (
            pynutil.delete(optional_prefix_boundary)
            + pynutil.insert("negative: \"true\" ")
            + pynutil.delete(negative_sign_chars)
            + pynutil.delete(pynini.closure(" ", 0, 1))
        ).optimize()
        leading_optional_negative_tag = pynini.closure(process_negative_sign, 0, 1)

        # --- Integer and Decimal Part Definitions --- (保持不变)
        number_string_with_commas = pynini.closure(pynini.union(NEMO_DIGIT, pynini.accep(","))).optimize()
        integer_part_verbalized = number_string_with_commas @ cardinal_numbers
        integer_part_tagged = (
            pynutil.insert("integer_part: \"") +
            integer_part_verbalized +
            pynutil.insert("\"")
        )
        decimal_part_verbalized = number_string_with_commas @ cardinal_numbers
        decimal_part_tagged = (
            pynini.accep(".")
            + pynutil.insert("decimal_part: \"") +
            decimal_part_verbalized +
            pynutil.insert("\"")
        )
        optional_decimal_part_tagged = pynini.closure(
            pynutil.insert(" ") + decimal_part_tagged, 0, 1
        )

        # --- Money Expression Structures --- (保持不变)
        prefix_form = (
            currency_component
            + pynutil.insert(" ")
            + integer_part_tagged
            + optional_decimal_part_tagged
        )
        suffix_form = (
            integer_part_tagged
            + optional_decimal_part_tagged
            + pynutil.insert(" ")
            + currency_component
        )
        money_amount_structure = (prefix_form | suffix_form).optimize()

        # --- NEW: Optional Suffix "分" ---
        # This will consume "分" after a money amount, and can optionally add a tag.
        # For now, we'll add a tag to make it explicit.
        optional_suffix_fun = pynini.closure(
            pynutil.insert(" ") + pynutil.insert("unit: \"分\"") + pynini.cross("分", ""), # Consume "分", add tag
            0, 1
        ).optimize()
        # Alternative: Just consume "分" without adding a tag to keep it simpler
        # optional_suffix_fun = pynini.closure(pynutil.delete("分"), 0, 1).optimize()
        # Let's go with adding a tag, it's more informative.

        # --- Final Graph Construction --- (MODIFIED)
        graph = (
            leading_optional_negative_tag
            + money_amount_structure
            + optional_suffix_fun # Add the optional suffix handler here
        )
        
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()