from parsing.ll1 import LL1


def test():
    s = '''
    
    2 * 3;
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'multiplication_expression',
        'content': {
            'left': {
                'type': 'number',
                'value': '2'
            },
            'right': {
                'type':'number',
                'value' : '3'
            }
        }
    }

    assert (ast == expected)
