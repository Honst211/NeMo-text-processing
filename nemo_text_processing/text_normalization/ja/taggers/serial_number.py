import pynini
from pynini.lib import pynutil
# Use the import path you provided
from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst

class SerialNumberFst(GraphFst):
    """
    FST for classifying general serial numbers, allowing interleaved letters, digits, and hyphens.
    Digits are read out individually. Hyphens are read as "の".
    Example:
        "YT123456789JP" -> serial_number { value: "YTいちにさんよんごろくななはちきゅうJP" }
        "LM2023-7745"   -> serial_number { value: "LMにゼロにさんのななななよんご" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="serial_number", kind="classify", deterministic=deterministic)

        # 数字到假名 (Digits to Kana)
        digit_map = {
            "0": "ゼロ",  # Using ゼロ is common for serial numbers
            "1": "いち",
            "2": "に",
            "3": "さん",
            "4": "よん",  # Preferred over し for clarity
            "5": "ご",
            "6": "ろく",
            "7": "なな",  # Preferred over しち for clarity
            "8": "はち",
            "9": "きゅう", # Preferred over く
        }
        digit_fst = pynini.string_map(digit_map.items()).optimize() # Added .optimize()

        # 字母保留原样（大小写均保留） (Letters pass through, case preserved)
        letters_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        letters = pynini.union(*[pynini.accep(c) for c in letters_str]).optimize() # Added .optimize()


        # 连字符 "-" 替换为 "の" (Hyphen "-" replaced with "の")
        hyphen_fst = pynini.cross("-", "の").optimize() # Added .optimize()

        # 逐字符处理单元 (Single character processing unit)
        char_fst = pynini.union(
            letters,
            digit_fst,
            hyphen_fst
        ).optimize() # Added .optimize()

        # 多字符拼接 (Concatenation of one or more characters)
        graph = pynini.closure(char_fst, 1).optimize() # Added .optimize()

        # 包装为 classify 格式 (Wrap in classify format for the value part)
        # Optimizing this intermediate FST before passing to add_tokens
        final_graph_value = (pynutil.insert('value: "') + graph + pynutil.insert('"')).optimize() # Added .optimize()

        # The add_tokens method from GraphFst (when kind="classify")
        # will further wrap this with 'serial_number { ... }'.
        self.fst = self.add_tokens(final_graph_value).optimize() # Final optimize was already here