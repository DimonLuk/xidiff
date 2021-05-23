Feature: approximate equation
    Basic scenario of approximating equations.

    Scenario: should approximate equation
        Given real part of equation, imaginary part of equation, boundary conditions and variable ranges
        And xidiff variables are initialized
        And xidiff equation is initialized
        And xidff solver is intia1ized

        When xidiff solver approximated the equation

        Then it is possible to evaluate model