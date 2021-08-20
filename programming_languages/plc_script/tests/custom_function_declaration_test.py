from parsing.ll1 import LL1


def test():
    s = '''
    function mix(a,b) {
        return a+b;
    }
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'function_declaration_statement',
        'name': 'mix',
        'arguments': [
            {'type': 'identifier',
             'value': 'a'
             },
            {'type': 'identifier',
             'value': 'b'
             }
        ],
        'content': {
            'type': 'block',
            'content': {
                    'type': 'return_statement',
                    'content':
                        {
                            'type': 'addition_expression',
                            'content': {
                                'left': {
                                    'type': 'identifier', 'value': 'a'},
                                'right': {
                                    'type': 'identifier', 'value': 'b'}
                            }
                        }
            }

        }
    }

    assert (ast == expected)
