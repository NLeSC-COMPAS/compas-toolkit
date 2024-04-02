using CompasToolkit

include("common.jl")

CompasToolkit.init_context(0)

A = rand(ComplexF32, 100, 90, 5)
B = rand(ComplexF32, 100, 90, 5)

sum_answer, diff_answer = CompasToolkit.compute_residual(A, B)

diff_expected = A - B
print_equals_check(diff_expected, diff_answer)

sum_expected = sum(abs.(diff_expected).^2)
print_equals_check([sum_expected], [sum_answer])
