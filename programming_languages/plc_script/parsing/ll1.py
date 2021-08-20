'''
LL(1) parser for plc_script
Author:
Goran Trlin
https://playandlearntocode.com
'''

from tokenization.tokenizer import Tokenizer


class LL1:
    s = ''
    _tokenizer = None
    _lookahead = None

    def __init__(self, s):
        self.s = s
        self._tokenizer = Tokenizer(s)

    def parse(self):
        self._lookahead = self._tokenizer.get_next_token()
        return self.program()

    def program(self):
        return self.statement_list()

    def take(self, token_type):
        if self._lookahead is None or self._lookahead['type'] != token_type:
            raise Exception('Expected ' + token_type + ' but got ' + (
                self._lookahead['type'] if self._lookahead is not None else 'None'))
        else:
            ret = self._lookahead
            self._lookahead = self._tokenizer.get_next_token()
            return ret

    def statement_list(self):
        st_arr = []

        left = self.statement()

        while self._lookahead is not None:
            if self._lookahead['type'] == 'block_close':
                break

            st = self.statement()
            st_arr.append(st)

        if len(st_arr) > 0:
            return [left] + st_arr
        else:
            return left

    def statement(self):
        if self._lookahead['type'] == 'keyword_if':
            ret = self.if_statement()
        elif self._lookahead['type'] == 'keyword_while':
            ret = self.while_statement()
        elif self._lookahead['type'] == 'block_open':
            ret = self.block_statement()
        elif self._lookahead['type'] == 'delimiter':
            ret = self.empty_statement()
        elif self._lookahead['type'] == 'keyword_function':
            ret = self.function_declaration_statement()
        elif self._lookahead['type'] == 'keyword_return':
            ret = self.return_statement()
        else:
            ret = self.expression_statement()

        return ret

    def return_statement(self):
        self.take('keyword_return')
        e = self.expression()
        self.take('delimiter')

        return {
            'type': 'return_statement',
            'content': e
        }

    def function_declaration_statement(self):
        self.take('keyword_function')
        name = self.identifier()
        self.take('parentheses_open')

        if self._lookahead['type'] != 'parentheses_close':
            args = self.arguments_declaration()

        self.take('parentheses_close')
        body = self.block_statement()

        return {
            'type': 'function_declaration_statement',
            'name': name['value'],
            'arguments': args,
            'content': body
        }

    def arguments_declaration(self):
        arg_list = []
        identifier = self.identifier()
        arg_list.append(identifier)
        while self._lookahead['type'] == 'comma':
            self.take('comma')
            identifier = self.identifier()
            arg_list.append(identifier)

        return arg_list

    def arguments_call(self):
        arg_list = []
        if self._lookahead['type'] == 'identifier':
            next = self.identifier()
        else:
            next = self.literal()

        arg_list.append(next)
        while self._lookahead['type'] == 'comma':
            self.take('comma')
            if self._lookahead['type'] == 'identifier':
                next = self.identifier()
            else:
                next = self.literal()
            arg_list.append(next)

        return arg_list

    def empty_statement(self):
        self.take('delimiter')
        return {
            'type': 'empty_statement'
        }

    def block_statement(self):
        self.take('block_open')
        content = self.statement_list()
        self.take('block_close')

        return {
            'type': 'block',
            'content': content
        }

    def if_statement(self):
        self.take('keyword_if')
        self.take('parentheses_open')
        condition = self.expression()
        self.take('parentheses_close')

        case_true = self.block_statement()
        case_false = None

        if self._lookahead['type'] == 'keyword_else':
            self.take('keyword_else')
            case_false = self.block_statement()

        return {
            'type': 'if_statement',
            'condition': condition,
            'case_true': case_true,
            'case_false': case_false
        }

    def while_statement(self):
        self.take('keyword_while')
        self.take('parentheses_open')
        condition = self.expression()
        self.take('parentheses_close')

        while_body = self.block_statement()

        return {
            'type': 'while_statement',
            'condition': condition,
            'while_body': while_body
        }

    def expression_statement(self):
        ret = self.expression()
        self.take('delimiter')
        return ret

    def assignment_expression(self):
        left = self.addition_subtraction_expression()
        if self._lookahead['type'] == 'assignment_operator':
            op = self.take('assignment_operator')
            right = self.addition_subtraction_expression()
            return {
                'type': 'assignment_expression',
                'content': {
                    'left': left,
                    'right': right
                }
            }
        else:
            return left

    def addition_subtraction_expression(self):
        left = self.multiplication_division_expression()
        cur = None

        operators = ['addition_operator', 'subtraction_operator']
        expression_operator_map = {'addition_operator': 'addition_expression',
                                   'subtraction_operator': 'subtraction_expression'}

        if self._lookahead['type'] not in operators:
            return left
        else:
            while self._lookahead['type'] in operators:
                cur_type = self._lookahead['type']
                op = self.take(cur_type)
                right = self.multiplication_division_expression()

                if cur is not None:
                    left = cur

                cur = {
                    'type': expression_operator_map[cur_type],
                    'content': {
                        'left': left,
                        'right': right
                    }
                }
            return cur

    def multiplication_division_expression(self):
        left = self.basic_expression()
        cur = None

        operators = ['multiplication_operator', 'division_operator', 'operator_equals', 'operator_less_than',
                     'operator_greater_than']
        expression_operator_map = {'multiplication_operator': 'multiplication_expression',
                                   'division_operator': 'division_expression',
                                   'operator_equals': 'operator_equals',
                                   'operator_less_than': 'operator_less_than',
                                   'operator_greater_than': 'operator_greater_than'
                                   }

        if self._lookahead['type'] not in operators:
            return left
        else:
            while self._lookahead['type'] in operators:
                cur_type = self._lookahead['type']
                op = self.take(cur_type)
                right = self.basic_expression()

                if cur is not None:
                    left = cur

                cur = {
                    'type': expression_operator_map[cur_type],
                    'content': {
                        'left': left,
                        'right': right
                    }
                }
            return cur

    def expression(self):
        left = self.assignment_expression()

        operators = ['assignment_operator', 'multiplication_operator', 'addition_operator']
        operations = {
            'assignment_operator': 'assignment_operation',
            'multiplication_operator': 'multiplication_operation',
            'addition_operator': 'addition_operation',
        }

        cur = None
        while self._lookahead is not None and (self._lookahead['type'] in operators):
            operator = self._lookahead['type']
            self.take(operator)
            right = self.assignment_expression()

            if cur is not None:
                left = cur

            cur = {
                'type': operations[operator],
                'left': left,
                'right': right
            }

        if cur is None:
            return left
        else:
            return cur

    def basic_expression(self):
        if self._lookahead['type'] == 'parentheses_open':
            return self.parentheses_expression()
        if self._lookahead['type'] == 'identifier':
            return self.function_call_expression()
        else:
            return self.literal()

    def parentheses_expression(self):
        self.take('parentheses_open')
        e = self.expression()
        self.take('parentheses_close')
        return e

    def function_call_expression(self):
        identifier = self.identifier()

        if self._lookahead['type'] == 'parentheses_open':
            self.take('parentheses_open')
            args = None
            if self._lookahead['type'] != 'parentheses_close':
                args = self.arguments_call()

            self.take('parentheses_close')
            return {
                'type': 'function_call_expression',
                'callee': identifier['value'],
                'arguments': args
            }
        else:
            return identifier

    def identifier(self):
        ret = self.take('identifier')
        return ret

    def literal(self):
        ret = None
        if self._lookahead['type'] == 'string':
            ret = self.string()
        elif self._lookahead['type'] == 'number':
            ret = self.number()
        return ret

    def number(self):
        ret = self.take('number')
        return ret

    def string(self):
        ret = self.take('string')
        ret['value'] = ret['value'][1:-1]
        return ret
