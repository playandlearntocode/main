from parsing.ll1 import LL1


def test():
    s = '''
    
    6 / 2;
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'division_expression',
        'content': {
            'left': {
                'type': 'number',
                'value': '6'
            },
            'right': {
                'type':'number',
                'value' : '2'
            }
        }
    }

    assert (ast == expected)
