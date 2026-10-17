# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Lars Hellemo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Plot benchmark results from bench/results/*.csv into docs/*.svg.
#
#   julia --project=bench bench/plots.jl
#
# Run bench/chapel/run.sh and bench/bench.jl first to produce the CSV files
# (bench/strings.jl adds the strings chart).

using AlgebraOfGraphics, CairoMakie, CSV, DataFrames

const RESULTS = joinpath(@__DIR__, "results")
const OUT = joinpath(@__DIR__, "..", "docs")

# Reference palette (categorical slots 1–3, in fixed order) and neutral inks.
const SURFACE = colorant"#fcfcfb"
const INK = colorant"#0b0b0b"
const INK2 = colorant"#52514e"
const GRID = colorant"#e6e5e0"
const SERIES = ["psort! (reused scratch)" => colorant"#2a78d6",
                "psort! (fresh scratch)" => colorant"#eb6834",
                "Chapel sort" => colorant"#1baf7a"]
const MARKERS = [:circle, :rect, :utriangle]

function load()
    df = vcat(CSV.read(joinpath(RESULTS, "julia.csv"), DataFrame),
              CSV.read(joinpath(RESULTS, "chapel.csv"), DataFrame))
    df.melems = df.n ./ df.seconds ./ 1e6
    return df
end

const AXIS = (; backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
              xgridwidth = 1, ygridwidth = 1, leftspinecolor = INK2,
              bottomspinecolor = INK2, topspinevisible = false,
              rightspinevisible = false, xtickcolor = INK2, ytickcolor = INK2,
              xticklabelcolor = INK2, yticklabelcolor = INK2, xlabelcolor = INK,
              ylabelcolor = INK, titlecolor = INK, titlealign = :left,
              titlesize = 17, subtitlecolor = INK2)

function scaling_plot(df, nmax)
    names = first.(SERIES)
    d = filter(r -> r.method in names, df)
    plt = data(d) *
          mapping(:ntasks => "Tasks", :melems => "Throughput (M elements/s)";
                  color = :method => sorter(names...) => "") *
          (visual(Lines; linewidth = 2) +
           mapping(marker = :method => sorter(names...) => "") *
           visual(Scatter; markersize = 11, strokewidth = 2, strokecolor = SURFACE))
    fig = draw(plt,
               scales(Color = (; palette = last.(SERIES)),
                      Marker = (; palette = MARKERS));
               axis = (; AXIS..., xscale = log2, xticks = [1, 2, 4, 8],
                       title = "Scaling: sorting 2^27 random UInt64",
                       subtitle = "Higher is better · Intel Core Ultra 7 268V, 4P + 4E cores",
                       limits = (0.85, nmax * 1.9, 0, nothing)),
               legend = (; position = :bottom, framevisible = false,
                         labelcolor = INK, tellheight = true),
               figure = (; size = (720, 480), backgroundcolor = SURFACE,
                         fontsize = 14))
    # Direct labels at the right-hand end of each series, in text ink.
    ax = fig.figure.content[1]
    for name in names
        r = only(eachrow(filter(r -> r.method == name && r.ntasks == nmax, d)))
        text!(ax, r.ntasks, r.melems; text = "  $(round(Int, r.melems))",
              align = (:left, :center), color = INK, fontsize = 13)
    end
    return fig
end

function methods_plot(df, nmax)
    labels = ["Base.sort!" => "Base.sort!",
              "chunk sort + parallel merge" => "chunk sort + merge",
              "AcceleratedKernels.sort!" => "AcceleratedKernels.sort!",
              "Chapel sort" => "Chapel sort",
              "psort! (fresh scratch)" => "psort! (fresh scratch)",
              "psort! (reused scratch)" => "psort! (reused scratch)"]
    d = filter(r -> r.ntasks == nmax && r.method in first.(labels), df)
    d.label = [Dict(labels)[m] for m in d.method]
    sort!(d, :melems)
    plt = data(d) *
          mapping(:label => sorter(d.label...) => "",
                  :melems => "Throughput (M elements/s)") *
          visual(BarPlot; direction = :x, color = last(SERIES[1]), gap = 0.35,
                 bar_labels = :y, label_formatter = v -> string(round(Int, v)),
                 label_color = INK, label_size = 13, label_offset = 6)
    fig = draw(plt;
               axis = (; AXIS..., ygridvisible = false,
                       title = "All methods at $nmax threads (2^27 UInt64)",
                       subtitle = "Higher is better",
                       limits = (0, maximum(d.melems) * 1.15, nothing, nothing)),
               figure = (; size = (720, 400), backgroundcolor = SURFACE,
                         fontsize = 14))
    return fig
end

# Speedup bars (Base or DataFrames time / ChapelSort time), one bar per row of
# `d` (columns :label, :speedup and :series), with a reference line at 1×.
function speedup_plot(d, labels, series, title, subtitle; size = (720, 480), facet = nothing)
    m = mapping(:label => sorter(labels...) => "", :speedup => "Speedup (×)";
                color = :series => sorter(series...) => "",
                dodge = :series => sorter(series...) => "")
    facet === nothing || (m = m * mapping(row = facet => sorter(unique(d[!, facet])...)))
    bars = data(d) * m *
           visual(BarPlot; direction = :x, gap = 0.3, dodge_gap = 0.06,
                  bar_labels = :y, label_formatter = v -> string(round(v, digits = 1), "×"),
                  label_color = INK, label_size = 12, label_offset = 4)
    ref = mapping([1.0]) * visual(VLines; color = INK2, linestyle = :dash, linewidth = 1)
    nseries = length(series)
    legend = nseries > 1 ? (; position = :bottom, framevisible = false, labelcolor = INK) :
             (; show = false)
    fig = draw(bars + ref,
               scales(Color = (; palette = last.(SERIES)[1:nseries]));
               axis = (; AXIS..., ygridvisible = false, title, subtitle,
                       limits = (0, maximum(d.speedup) * 1.18, nothing, nothing)),
               facet = (; linkyaxes = :none),
               legend,
               figure = (; size, backgroundcolor = SURFACE, fontsize = 14))
    return fig
end

function strings_plot()
    r = CSV.read(joinpath(RESULTS, "strings.csv"), DataFrame)
    w = unstack(r, [:case, :op], :method, :seconds)
    w.speedup = w.Base ./ w.ChapelSort
    w.label = w.case
    w.series = ifelse.(w.op .== "sort", "psort! vs sort!", "psortperm vs sortperm")
    labels = reverse(unique(w.label))
    n = first(r.n)
    return speedup_plot(w, labels, ["psort! vs sort!", "psortperm vs sortperm"],
                        "Sorting 10^7 values: speedup over Base",
                        "$(Int(first(r.nthreads))) threads · higher is better · dashed line = Base";
                        size = (760, 560))
end

function main()
    df = load()
    nmax = maximum(df.ntasks)
    mkpath(OUT)
    save(joinpath(OUT, "scaling.svg"), scaling_plot(df, nmax))
    save(joinpath(OUT, "methods.svg"), methods_plot(df, nmax))
    isfile(joinpath(RESULTS, "strings.csv")) &&
        save(joinpath(OUT, "strings.svg"), strings_plot())
    println("wrote plots to ", OUT)
end

main()
