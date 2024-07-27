function unique_data(data)
    return map(x -> add_counts(data, x), unique(data))
end

function add_counts(data, u)
    N = count(x -> x == u, data)
    return (u..., N = N)
end
