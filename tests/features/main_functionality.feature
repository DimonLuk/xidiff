Feature: approximate equation
    Basic scenario of approximating equations.

    Scenario: should approximate equation
        Given real part, imaginary part, boundary conditions and variable ranges of <equation>
        And xidiff variables are initialized
        And xidiff equation is initialized
        And xidff solver is intia1ized

        When xidiff solver approximated the equation

        Then it is possible to evaluate model
        And it is possible to save model
        And it is possible to restore model
        And it is possible to evaluate model with the same results
        And clean up

        Examples:
        | equation   |
        | equation_0 |
        | equation_1 |
        | equation_2 |