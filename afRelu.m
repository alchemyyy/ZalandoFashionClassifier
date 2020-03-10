function a = afRelu(n)
a = zeros(length(n),1);
for i=1:length(n)
    if (n(i) > 0)
        a(i) = n(i);
    else
        a(i) = n(i) * .001;
    end
end
end

