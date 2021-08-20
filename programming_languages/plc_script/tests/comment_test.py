from parsing.ll1 import LL1


def test():
    s = '''
    
    // this a  test comment
    2  ;
    
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'number',
        'value': '2'
    }

    assert (ast == expected)
