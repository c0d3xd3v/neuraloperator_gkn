from ngsolve import *

def solvePoission(fes, gfu, g=CF(1.), c=CF(1.)):
    # define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # the right hand side
    f = LinearForm(fes)
    f += g * v * dx
    # the bilinear-form 
    a = BilinearForm(fes, symmetric=True)
    a += c*grad(u)*grad(v)*dx

    a.Assemble()
    f.Assemble()

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * res

    return gfu
