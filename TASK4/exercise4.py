# CODTECH Internship Task-4
# Optimization Model using PuLP

from pulp import LpMaximize, LpProblem, LpVariable

# Create optimization problem
model = LpProblem(name="factory-production", sense=LpMaximize)

# Decision variables
A = LpVariable(name="Product_A", lowBound=0)
B = LpVariable(name="Product_B", lowBound=0)

# Objective function (maximize profit)
model += 20 * A + 30 * B

# Constraints
model += 2 * A + 1 * B <= 40   # Machine hours
model += 1 * A + 2 * B <= 60   # Labor hours

# Solve the problem
model.solve()

# Print results
print("Optimal Production Plan")
print("Product A:", A.value())
print("Product B:", B.value())

print("Maximum Profit:", model.objective.value())