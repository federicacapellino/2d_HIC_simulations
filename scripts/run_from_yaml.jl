#!/usr/bin/env julia
using YAML
using Logging

cfg_path = joinpath(@__DIR__, "config.yaml")
if !isfile(cfg_path)
    @error "Config file not found" path=cfg_path
    exit(1)
end

cfg = YAML.load_file(cfg_path)

physics = get(cfg, "physics parameter", Dict())
runp = get(cfg, "run parameter", Dict())

function assign_dict(d::AbstractDict)
    for (k, v) in d
        try
            s = Symbol(k)
            @eval Main $(s) = $(v)
            @info "Set parameter" key=k value=v
        catch e
            @warn "Failed to set parameter" key=k err=String(e)
        end
    end
end

assign_dict(runp)
assign_dict(physics)

include(joinpath(@__DIR__, "collision_event.jl"))
