#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class AddressNumberFst(GraphFst):
    """
    Finite state transducer for classifying address numbers, applying specific Japanese reading rules.
    - Segments before the last are read in Kanji style (e.g., "10" -> "十").
    - The final segment (room/bldg number) is read based on the presence of '0':
      - If it contains a '0', it is read digit-by-digit, with '0' read as 'マル' (e.g., "809" -> "ハチマルキュー").
      - If it does not contain a '0', it is read in Kanji style (e.g., "21" -> "二十一").
    - Postal codes (e.g., 〒123-4567) are read digit-by-digit, with '0' read as 'ゼロ'.
    """

    _ties_map_kanji_fb_static = {str(i): k for i, k in enumerate(["", "十", "二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十"])}
    _digit_map_kanji_fb_static = {str(i): k for i, k in enumerate(["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"])}

    def __init__(self, deterministic: bool = True):
        super().__init__(name="address_number", kind="classify", deterministic=deterministic)

        self._ties_map_kanji_fallback = AddressNumberFst._ties_map_kanji_fb_static
        self._digit_map_kanji_fallback = AddressNumberFst._digit_map_kanji_fb_static
        
        try:
            self.graph_digit_file = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            self.graph_zero_from_file = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
            self.graph_ties_file = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        except Exception as e:
            print(f"Warning: Could not load number data files ('{get_abs_path('data/numbers/digit.tsv')}', etc. Error: {e}). Using simplified fallbacks.")
            self.graph_digit_file = pynini.string_map(self._digit_map_kanji_fallback).optimize()
            self.graph_zero_from_file = pynini.cross("0", "零").optimize()
            self.graph_ties_file = pynini.string_map(self._ties_map_kanji_fallback).optimize()


        self.no_zero_digit_accepter = pynini.difference(NEMO_DIGIT, pynini.accep("0")).optimize()
        self.separator = (pynini.cross("-", "の") | pynini.cross("—", "の")).optimize()

        self.kanji_prefix_for_hyaku_sen = (
            pynini.difference(NEMO_DIGIT, pynini.union("0", "1")).optimize() @ self.graph_digit_file
        ).optimize()

        _default_style_digit_map = { "1": "イチ", "2": "ニー", "3": "サン", "4": "ヨン", "5": "ゴ", "6": "ロク", "7": "ナナ", "8": "ハチ", "9": "キュー"}
        
        # Default digit reader (for postal codes, etc.), reads '0' as 'ゼロ'
        _sdsr_rules = [pynini.cross(k, v) for k, v in _default_style_digit_map.items()]
        _sdsr_rules.append(pynini.cross("0", "ゼロ"))
        self.default_style_digit_sequence_reader = pynini.closure(pynini.union(*_sdsr_rules), 1).optimize()
        
        # START OF MODIFICATION
        # Specialized digit reader for addresses, reads '0' as 'マル'
        # This is used for room/bldg numbers like "809" -> "ハチマルキュー"
        _address_style_rules = [pynini.cross(k, v) for k, v in _default_style_digit_map.items()]
        _address_style_rules.append(pynini.cross("0", "マル"))
        self.address_style_digit_sequence_reader = pynini.closure(pynini.union(*_address_style_rules), 1).optimize()
        # END OF MODIFICATION

        self.kanji_style_multiple_zeros = pynini.union(
            pynini.cross("000", "零零零"), pynini.cross("00", "零零"), self.graph_zero_from_file
        ).optimize()

        def process_kanji_number_func():
            # This complex function correctly creates a grammar for traditional Japanese number reading.
            # It remains unchanged as its core logic for converting numbers like "21" to "二十一" is correct.
            kanji_0_to_9 = self.graph_digit_file
            kanji_1_to_9 = (self.no_zero_digit_accepter @ self.graph_digit_file).optimize()
            
            def get_mapped_output_string(input_str: str, mapping_fst: pynini.Fst) -> str:
                res_str = ""
                try:
                    composed_fst = pynini.compose(pynini.accep(input_str), mapping_fst).optimize()
                    if composed_fst.num_states() > 0 and composed_fst.start() != pynini.NO_STATE_ID:
                        res_str = composed_fst.string(output_token_type="utf8")
                except pynini.FstOpError: pass 
                except Exception: pass
                return res_str

            output_for_one_from_ties = get_mapped_output_string("1", self.graph_ties_file)
            rule_10 = pynini.cross("10", "十") 
            rule_11_to_19 = pynutil.delete("1") + pynutil.insert("十") + kanji_1_to_9 

            rules_for_X0_list = []
            for i in range(2, 10):
                digit_str = str(i)
                kanji_tens_output = get_mapped_output_string(digit_str, self.graph_ties_file)
                if kanji_tens_output: rules_for_X0_list.append(pynini.cross(digit_str + "0", kanji_tens_output))
                else:
                    fallback_tens = self._ties_map_kanji_fallback.get(digit_str)
                    if fallback_tens: rules_for_X0_list.append(pynini.cross(digit_str + "0", fallback_tens))
            rule_20_to_90_exact = pynini.union(*rules_for_X0_list) if rules_for_X0_list else pynini.accep("").optimize()

            rules_for_XY_list = []
            for i in range(2, 10):
                digit_str = str(i)
                tens_prefix_fst_candidate = None
                output_str_for_tens_prefix = get_mapped_output_string(digit_str, self.graph_ties_file)
                if output_str_for_tens_prefix: tens_prefix_fst_candidate = pynini.cross(digit_str, output_str_for_tens_prefix)
                else: 
                    fallback_tens_str = self._ties_map_kanji_fallback.get(digit_str)
                    if fallback_tens_str: tens_prefix_fst_candidate = pynini.cross(digit_str, fallback_tens_str)
                if tens_prefix_fst_candidate: rules_for_XY_list.append(tens_prefix_fst_candidate + kanji_1_to_9)
            rule_21_to_99_compound = pynini.union(*rules_for_XY_list) if rules_for_XY_list else pynini.accep("").optimize()
            
            graph_0_99 = pynini.union(kanji_0_to_9, rule_10, rule_11_to_19, rule_20_to_90_exact, rule_21_to_99_compound).optimize()
            rule_X00 = ((pynini.cross("1", "百") | (self.kanji_prefix_for_hyaku_sen + pynutil.insert("百"))) + pynutil.delete("00")).optimize()
            yy_part_not_00_consumer = (pynini.difference(NEMO_DIGIT**2, pynini.accep("00")).optimize() @ graph_0_99)
            rule_XYY_general = ((pynini.cross("1", "百") | (self.kanji_prefix_for_hyaku_sen + pynutil.insert("百"))) + yy_part_not_00_consumer).optimize()
            graph_100_to_999_raw = pynini.union(rule_X00, rule_XYY_general).optimize()
            graph_100_to_999 = ((NEMO_DIGIT ** 3) @ graph_100_to_999_raw).optimize()
            rule_X000 = ((pynini.cross("1", "千") | (self.kanji_prefix_for_hyaku_sen + pynutil.insert("千"))) + pynutil.delete("000")).optimize()
            graph_0_to_999_for_thousands_yyy_part = pynini.union(graph_100_to_999_raw, graph_0_99).optimize() 
            yyy_part_not_000_consumer = (pynini.difference(NEMO_DIGIT**3, pynini.accep("000")).optimize() @ graph_0_to_999_for_thousands_yyy_part)
            rule_XYYY_general = ((pynini.cross("1", "千") | (self.kanji_prefix_for_hyaku_sen + pynutil.insert("千"))) + yyy_part_not_000_consumer).optimize()
            graph_1000_to_9999_raw = pynini.union(rule_X000, rule_XYYY_general).optimize()
            graph_1000_to_9999 = ((NEMO_DIGIT ** 4) @ graph_1000_to_9999_raw).optimize()
            leading_zeros_with_kanji_suffix = pynini.union(pynini.cross("00", "零零") + kanji_1_to_9, pynini.cross("0", "零") + kanji_1_to_9).optimize()
            final_kanji_graph = pynini.union(graph_1000_to_9999, graph_100_to_999, graph_0_99, leading_zeros_with_kanji_suffix, self.kanji_style_multiple_zeros).optimize()
            return final_kanji_graph

        kanji_style_number = process_kanji_number_func()

        # Add constraint: address numbers should not start with 0
        # This helps avoid conflicts with telephone numbers
        non_zero_start_digit = pynini.difference(NEMO_DIGIT, pynini.accep("0")).optimize()
        
        # Create a pattern for numbers that don't start with 0
        # This applies to all address number segments except postal codes
        non_zero_kanji_number = pynini.compose(
            non_zero_start_digit + pynini.closure(NEMO_DIGIT, 0),
            kanji_style_number
        ).optimize()

        # Define logic for the final segment of an address (e.g., room/bldg number)
        # This segment's reading style depends on whether it contains a "0"

        # 1. An acceptor for any sequence of digits that does NOT contain "0" AND doesn't start with "0"
        digits_no_zero_seq = pynini.closure(self.no_zero_digit_accepter, 1).optimize()

        # 2. An acceptor for any sequence of digits that MUST contain at least one "0" BUT doesn't start with "0"
        digits_with_zero_seq = pynini.compose(
            non_zero_start_digit + pynini.closure(NEMO_DIGIT, 0),
            pynini.difference(pynini.closure(NEMO_DIGIT, 1), digits_no_zero_seq)
        ).optimize()

        # 3. Rule for final segments WITHOUT "0": apply kanji-style reading
        # e.g., "21" -> "二十一"
        final_segment_no_zero = digits_no_zero_seq @ kanji_style_number

        # START OF MODIFICATION
        # 4. Rule for final segments WITH "0": apply special address-style digit-by-digit reading ('0' -> 'マル')
        # e.g., "809" -> "ハチマルキュー" (but "009" would be rejected as it starts with 0)
        final_segment_with_zero = digits_with_zero_seq @ self.address_style_digit_sequence_reader
        # END OF MODIFICATION

        # 5. Combine them into a single rule for the final segment
        final_segment_logic = pynini.union(final_segment_no_zero, final_segment_with_zero).optimize()
        
        # Address patterns using the new logic with non-zero start constraint
        # All segments except the last use non-zero kanji-style. The last uses the conditional final_segment_logic.
        pattern1 = (non_zero_kanji_number + self.separator + non_zero_kanji_number + self.separator + non_zero_kanji_number + self.separator + final_segment_logic).optimize()
        pattern2 = (non_zero_kanji_number + self.separator + non_zero_kanji_number + self.separator + final_segment_logic).optimize()
        pattern3 = (non_zero_kanji_number + self.separator + final_segment_logic).optimize()

        # Postal code pattern (unchanged, as it was correct)
        # It correctly uses the 'default_style_digit_sequence_reader' which reads '0' as 'ゼロ'.
        prefix_to_keep_for_postal = (
            pynini.cross("郵便番号は", "郵便番号は") |       
            pynini.cross("郵便番号:", "郵便番号:") |       
            pynini.cross("郵便番号：", "郵便番号： ") |     
            pynini.cross("郵便番号 ", "郵便番号") |     
            pynini.cross("郵便番号", "郵便番号") |
            pynini.cross("郵便は", "郵便は") |           
            pynini.cross("〒", "〒 ") 
        ).optimize()

        postal_first_part = ((NEMO_DIGIT ** 3) @ self.default_style_digit_sequence_reader)
        postal_second_part = ((NEMO_DIGIT ** 4) @ self.default_style_digit_sequence_reader)
        
        pattern4_postal_code = (
            pynini.cross(pynini.union("〒", "郵便番号"), "郵便番号") + pynini.closure(pynutil.delete(" "), 0, 1) +
            pynini.closure(pynutil.delete(pynini.union(":", "：")), 0, 1) + pynini.closure(pynutil.delete(" "), 0, 1) +
            postal_first_part + 
            self.separator + 
            postal_second_part
        ).optimize()
        
        # Final union of all address patterns
        # Note: final_segment_logic already has the non-zero start constraint built in
        number_part = pynini.union(
            pattern4_postal_code,
            pattern1, 
            pattern2, 
            pattern3,
            final_segment_logic # Allows a single number segment, e.g. for a simple building number (non-zero start)
        ).optimize()

        final_graph = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()