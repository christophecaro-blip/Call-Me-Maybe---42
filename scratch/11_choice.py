from src.decoding.constraints import ChoiceConstraint

c = ChoiceConstraint(options=["fn_add", "fn_add_numbers", "fn_greet"])

print(c.accepts("fn"), c.accepts("Hello"))       # attendu : True False
c.advance("fn_add")
print(c.accepts("_numbers"), c.accepts('"'))     # attendu : True True
print(c.accepts('",'), c.is_complete())          # attendu : False False
c.advance('"')
print(c.is_complete(), c.generated)              # attendu : True fn_add"
