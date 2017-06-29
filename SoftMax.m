function p = SoftMax (o)
    p = exp(o)/dot(ones(size(o)),exp(o));
end