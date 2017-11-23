from random import sample

class CSP():
    """
    class representing a csp problem with variables, their domains and constraints
    variables takes a dict with variable name : domain pairs
    constrains takes a dict with constraint functions as keys and a list of
    dependencies in the right order as values
    """
    def __init__(self, variables, constraints_dict):
        self.variables = list(variables.keys())
        self.domain_dict = variables
        self.constraints_dict= constraints

    def check_constraints(self, var_dict):
        """
        function checking all constraints with their dependencies as defined in
        constraint_args and values taken from the external var_dict.
        If no value is assigned to a variable yet in var_dict, None is assumend.
        """
        for constraint in self.constraints_dict.keys():
            #building args for constraint to be checked from var_dict
            args = self.constraints_dict[constraint]
            arg_values = []
            for arg in args:
                if arg in var_dict.keys():
                    arg_values.append(var_dict[arg])
                else:
                    arg_values.append(None)
            #if constraint in not sattisfied return False. no need to check others
            if not constraint(arg_values):
                return False
        #return True if all constraints are sattisfied
        return True

def constraint(args):
    A = args[0]
    B = args[1]
    C = args[2]
    if A==None or B==None or C==None:
        return True
    else:
        return A - C == 9*B

variables = {'A': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'B': [0, 1], \
                            'C':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
constraints = {constraint:['A', 'B', 'C']}
cryptarith = CSP(variables, constraints)

vars = {'A':10, 'B':10, 'C':10}

def rec_backtrack_search(assignment, not_assigned, csp):
    """
    is a recursive function performing backtrack search on an object of class CSP
    """
    #checking wheather every variable got assigned
    if not not_assigned:
        return assignment

    var = not_assigned.pop()
    #sampling a random order of variable's domain to get different results
    for value in sample(csp.domain_dict[var], len(csp.domain_dict[var])):
        assignment[var] = value
        if csp.check_constraints(assignment):
            result = rec_backtrack_search(assignment, not_assigned, csp)
            #back propagating of a solution through recursive calls
            if result != 'failure':
                return result
    # returning failure if branch does not contain a solution
    return 'failure'

def backtrack(csp):
    not_assigned = csp.variables
    return rec_backtrack_search({}, not_assigned, csp)

print(backtrack(cryptarith))
