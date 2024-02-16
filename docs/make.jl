using CoupledIsingModels
using Documenter

DocMeta.setdocmeta!(CoupledIsingModels, :DocTestSetup, :(using CoupledIsingModels); recursive=true)

makedocs(;
    modules=[CoupledIsingModels],
    authors="Gabriel Diaz Iturry <gabriel.diaz.iturry@gmail.com>",
    sitename="CoupledIsingModels.jl",
    format=Documenter.HTML(;
        canonical="https://gabo-di.github.io/CoupledIsingModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gabo-di/CoupledIsingModels.jl",
    devbranch="main",
)
