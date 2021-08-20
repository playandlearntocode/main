from parsing.ll1 import LL1


def test():
    s = '''
    a == 2;    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'operator_equals',
        'content': {
            'left': {
                'type': 'identifier',
                'value': 'a'
            },
            'right': {
                'type': 'number',
                'value': '2'
            }
        }
    }

    assert (ast == expected)
