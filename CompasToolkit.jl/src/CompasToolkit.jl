module CompasToolkit
  using CxxWrap
  @wrapmodule(joinpath("/home/stijn/projects/compas-toolkit/lib", "libcompas-julia.so"))

  function __init__()
    @initcxx
  end
end
