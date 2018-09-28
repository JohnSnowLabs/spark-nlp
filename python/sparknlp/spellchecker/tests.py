from data_prepare import ComplexToken

# test complex token handling
ct = ComplexToken(set(['ac', 'dc', '/', '-', '110v']))
assert(ct.belongs('ac/dc'))
assert(ct.belongs('ac-dc'))

assert(ct.parse('ac/dc-110v') == ['ac', '/', 'dc', '-', '110v'])
