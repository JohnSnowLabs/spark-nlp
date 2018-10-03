from data_prepare import ComplexToken, Percentage, SuffixedToken

# test complex token handling
ct = ComplexToken(set(['ac', 'dc', '/', '-', '110v']))
assert(ct.belongs('ac/dc'))
assert(ct.belongs('ac-dc'))

assert(ct.parse('ac/dc-110v') == ['ac', '/', 'dc', '-', '110v'])

# test percentage handling
per = Percentage()
assert(per.belongs('55%'))
assert(per.belongs('55.5%'))
assert(per.belongs('55.15%'))

# test suffixes handling
suff = SuffixedToken()
assert(suff.belongs('Patient:'))
assert(suff.belongs('12,'))
assert(suff.get_rep('pain"') == ['pain', '"'])