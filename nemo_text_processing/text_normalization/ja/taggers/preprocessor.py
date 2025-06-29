import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_SIGMA, NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class PreProcessorFst(GraphFst):
    '''
    Preprocessing of TN for Japanese text:
        1. fullwidth -> halfwidth char conversion (including spaces)
        2. Japanese quotation marks conversion
        3. Japanese punctuation conversion
        4. Replace spaces not between numbers with <|space|>
    Examples:
    ＡＢＣ -> ABC
    １２３ -> 123
    「こんにちは」 -> "こんにちは"
    。、 -> .,
    　 ->   (fullwidth space to halfwidth space)
    A B -> A<|space|>B
    1 2 -> 1 2 (space preserved between numbers)
    '''

    def __init__(
        self,
        fullwidth_to_halfwidth: bool = True,
    ):
        super().__init__(name="PreProcessor", kind="processor")
        
        if fullwidth_to_halfwidth:
            try:
                # 加载映射文件
                with open(get_abs_path('data/char/fullwidth_to_halfwidth.tsv'), 'r', encoding='utf-8') as f:
                    pairs = [line.strip().split(' ') for line in f if line.strip()]
                
                # 创建转换规则
                rules = []
                for source, target in pairs:
                    rule = pynini.cdrewrite(
                        pynini.cross(source, target),
                        "",
                        "",
                        NEMO_SIGMA
                    )
                    rules.append(rule)
                
                # 组合所有规则
                if rules:
                    self.fst = rules[0]
                    for rule in rules[1:]:
                        self.fst = pynini.compose(self.fst, rule).optimize()
                else:
                    self.fst = pynini.cdrewrite("", "", "", NEMO_SIGMA).optimize()
                    
            except Exception as e:
                print(f"Error loading mapping files: {str(e)}")
                self.fst = pynini.cdrewrite("", "", "", NEMO_SIGMA).optimize()
        else:
            self.fst = pynini.cdrewrite("", "", "", NEMO_SIGMA).optimize()

        # 创建空格替换的FST
        space_to_marker = pynini.cross(" ", "<|space|>")
        number_space_to_hyphen = pynini.cross(" ", "-")
        
        # 创建数字间空格替换为连字符的规则（优先级更高）
        number_space_rule = pynini.cdrewrite(
            number_space_to_hyphen,  # 将空格替换为连字符
            NEMO_DIGIT,     # 左侧必须是数字
            NEMO_DIGIT,     # 右侧必须是数字
            NEMO_SIGMA      # 允许的字符集
        ).optimize()
        
        # 创建其他空格替换规则（优先级更低）
        other_space_rule = pynini.cdrewrite(
            space_to_marker,  # 替换空格为标记
            "",              # 左侧上下文（任意）
            "",              # 右侧上下文（任意）
            NEMO_SIGMA      # 允许的字符集
        ).optimize()
        
        # 创建全角空格到半角空格的转换规则
        fullwidth_space_rule = pynini.cdrewrite(
            pynini.cross("　", " "),
            "",
            "",
            NEMO_SIGMA
        ).optimize()
        
        # 组合所有规则
        self.fst = pynini.compose(
            self.fst,
            fullwidth_space_rule
        ).optimize()
        
        self.fst = pynini.compose(
            self.fst,
            pynini.compose(number_space_rule, other_space_rule)
        ).optimize() 