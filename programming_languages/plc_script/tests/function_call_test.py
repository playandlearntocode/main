from parsing.ll1 import LL1


def test():
    s = '''
    mix(2,3);
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'function_call_expression',
        'callee': 'mix',
        'arguments': [
            {'type': 'number',
             'value': '2'
             },
            {'type': 'number',
             'value': '3'
             }
        ]
    }

    assert (ast == expected)
