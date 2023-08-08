module CompasToolkit
  using CxxWrap
  @wrapmodule(joinpath("/var/scratch/sheldens/compas-toolkit/lib", "libcompas-julia.so"))

  function __init__()
    @initcxx
  end
end
