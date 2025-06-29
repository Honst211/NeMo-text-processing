# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Japanese telephone numbers.
    
    Japanese telephone numbers typically follow these patterns:
    - Fixed line: 03-1234-5678, 0261-72-3456 (area code-local number)
    - Mobile: 090-1234-5678, 080-1234-5678, 070-1234-5678
    - Toll-free: 0120-123-456, 0800-123-456
    - With country code: +81-3-1234-5678, +81-90-1234-5678
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        # TTS-friendly Japanese digit mapping for telephone numbers
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
            "9": "キュー",
        }
        
        digit = pynini.union(*[pynini.cross(d, reading) for d, reading in tts_digit_map.items()])

        # Define a consistent separator that replaces hyphens, spaces, or dots with 'の'
        separator = pynini.union(pynini.cross("-", "の"), pynini.cross(" ", "の"), pynini.cross(".", "の"))

        # Helper to handle four digits (no special logic needed for now, can be expanded)
        four_digits = digit + digit + digit + digit

        # Variable length digit groups
        one_digit = digit
        two_digits = digit + digit
        three_digits = digit + digit + digit
        
        # Mobile patterns (0X0-XXXX-XXXX)
        mobile_prefixes = pynini.union(
            pynini.cross("050", "ゼロゴゼロ"),
            pynini.cross("070", "ゼロナナゼロ"),
            pynini.cross("080", "ゼロハチゼロ"),
            pynini.cross("090", "ゼロキューゼロ"),
        )
        mobile_patterns = mobile_prefixes + separator + four_digits + separator + four_digits
        
        # Telephone prompts in Japanese
        telephone_prompts = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        
        # International country code handling
        country_codes = pynini.union(
            pynini.cross("+81", "プラスハチイチ"),
            pynini.cross("+1", "プラスイチ"),
            pynini.cross("+44", "プラスヨンヨン"),
            pynini.cross("+86", "プラスハチロク"),
            # Generic pattern for other countries: + and 1-3 digits
            pynini.cross("+", "プラス") + digit + pynini.closure(digit, 0, 2),
        )
        
        country_code_graph = (
            pynini.closure(telephone_prompts + pynutil.delete(" "), 0, 1) + country_codes
        )
        country_code = pynutil.insert("country_code: \"") + country_code_graph + pynutil.insert("\"")
        country_code = country_code + pynini.closure(pynutil.delete("-"), 0, 1) + pynutil.insert(" ")
        
        # Parentheses pattern for area code, e.g., (03) -> のゼロサンの
        paren_pattern = pynutil.delete("(") + digit @ pynini.union(two_digits, three_digits) + pynutil.delete(")")
        paren_pattern = pynutil.insert("の") + paren_pattern + pynutil.insert("の")

        # Fixed line patterns (10 digits total)
        # 0X-XXXX-XXXX
        one_digit_area = pynini.cross("0", "ゼロ") + digit + separator + four_digits + separator + four_digits
        
        # 0XX-XXX-XXXX
        two_digit_area = pynini.cross("0", "ゼロ") + two_digits + separator + three_digits + separator + four_digits
        
        # 0XXX-XX-XXXX (Corrected pattern for cases like 0261-72-3456)
        three_digit_area = pynini.cross("0", "ゼロ") + three_digits + separator + two_digits + separator + four_digits
        
        # 0XXXX-X-XXXX
        four_digit_area = pynini.cross("0", "ゼロ") + four_digits + separator + one_digit + separator + four_digits

        # Combine all fixed line patterns
        fixed_patterns = pynini.union(one_digit_area, two_digit_area, three_digit_area, four_digit_area)

        # Free dial patterns (0120-XXX-XXX, etc.)
        free_dial_prefixes = pynini.union(
            pynini.cross("0120", "ゼロイチニゼロ"),
            pynini.cross("0800", "ゼロハチゼロゼロ"),
            pynini.cross("0570", "ゼロゴナナゼロ"), # Navidial
        )
        free_dial_patterns = free_dial_prefixes + separator + three_digits + separator + three_digits
        
        # Local numbers without area code (less common)
        non_zero_digit = pynini.union(*[pynini.cross(str(i), tts_digit_map[str(i)]) for i in range(1, 10)])
        local_patterns = pynini.union(
            non_zero_digit + three_digits + separator + four_digits, # XXXX-XXXX
            non_zero_digit + two_digits + separator + four_digits,  # XXX-XXXX
            non_zero_digit + one_digit + separator + four_digits,   # XX-XXXX
        )
        
        # Combine all domestic patterns
        domestic_numbers = pynini.union(mobile_patterns, fixed_patterns, free_dial_patterns, local_patterns)
        number_part_domestic = pynutil.insert("number_part: \"") + domestic_numbers + pynutil.insert("\"")
        
        # International patterns (country code already stripped)
        # Using a flexible pattern for numbers after the country code
        intl_number_group = pynini.union(one_digit, two_digits, three_digits, four_digits)
        international_numbers = (
            intl_number_group + pynini.closure(separator + intl_number_group, 1, 3)
        )
        number_part_international = pynutil.insert("number_part: \"") + international_numbers + pynutil.insert("\"")

        # Extension handling (内線 - naisen in Japanese)
        extension_prompt = pynini.union("内線", "内")
        extension_body = pynini.closure(digit, 1, 4)
        extension = (
            pynutil.insert("extension: \"")
            + pynini.cross(extension_prompt, "")
            + extension_body
            + pynutil.insert("\"")
        )
        extension = pynini.closure(pynutil.insert(" ") + extension, 0, 1)

        # Complete telephone number patterns
        graph = pynini.union(
            # International numbers with country code
            country_code + number_part_international + extension,
            country_code + number_part_international,
            # Domestic numbers
            number_part_domestic + extension,
            number_part_domestic,
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()