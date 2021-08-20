'''
Tokenizer for plc_script language
Author:
Goran Trlin
https://playandlearntocode.com
'''

import re
class Tokenizer:
    s = ''
    cursor = 0

    def __init__(self, s):
        self.s = s

    skip_tokens = {
        'whitespace': '\s+',
        'comment': '//.*?\n'
    }

    tokens = {
        'delimiter': ';',
        'number': '\d+',
        'string':'".*"',
        'block_open': '{',
        'block_close': '}',
        'operator_equals': '==',
        'operator_less_than': '<',
        'operator_greater_than': '>',
        'assignment_operator': '=',
        'subtraction_operator': '\-',
        'addition_operator': '\+',
        'multiplication_operator': '\*',
        'division_operator': '\/',
        'comma': ',',
        'parentheses_open': '\(',
        'parentheses_close': '\)',

        # keywords:
        'keyword_function': 'function',
        'keyword_return': 'return',
        'keyword_if': 'if',
        'keyword_else': 'else',
        'keyword_while': 'while',

        'identifier': '[a-zA-Z0-9]+'
    }

    def get_next_token(self):
        continue_skipping = True
        while continue_skipping:
            continue_skipping = False
            for token_name in self.skip_tokens:
                pattern = re.compile(self.skip_tokens[token_name])
                match = pattern.match(self.s, self.cursor)

                if match is not None:
                    self.cursor+= len(match[0])
                    continue_skipping = True

        for token_name in self.tokens:
            pattern = re.compile(self.tokens[token_name])
            match = pattern.match(self.s, self.cursor)

            if match is not None:
                self.cursor+= len(match[0])

                return {
                    'type': token_name,
                    'value': match[0]
                }

        return None