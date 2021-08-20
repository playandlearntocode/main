from parsing.ll1 import LL1


def test():
    s = '''

    a = 2;
    b = 3;
    
    function f1(a,b){
        return a*b;
    }
    
    c = f1(a,b);

    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = [{'type': 'assignment_expression',
                 'content': {'left': {'type': 'identifier', 'value': 'a'}, 'right': {'type': 'number', 'value': '2'}}},
                {'type': 'assignment_expression',
                 'content': {'left': {'type': 'identifier', 'value': 'b'}, 'right': {'type': 'number', 'value': '3'}}},
                {'type': 'function_declaration_statement', 'name': 'f1',
                 'arguments': [{'type': 'identifier', 'value': 'a'}, {'type': 'identifier', 'value': 'b'}],
                 'content': {'type': 'block', 'content': {'type': 'return_statement',
                                                          'content': {'type': 'multiplication_expression', 'content': {
                                                              'left': {'type': 'identifier', 'value': 'a'},
                                                              'right': {'type': 'identifier', 'value': 'b'}}}}}},
                {'type': 'assignment_expression', 'content': {'left': {'type': 'identifier', 'value': 'c'},
                                                              'right': {'type': 'function_call_expression',
                                                                        'callee': 'f1', 'arguments': [
                                                                      {'type': 'identifier', 'value': 'a'},
                                                                      {'type': 'identifier', 'value': 'b'}]}}}]

    assert (ast == expected)
