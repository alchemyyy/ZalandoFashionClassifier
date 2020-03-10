function a = afdRelu(n)
a = zeros(length(n),1);
coder.unroll();
for i=1:length(n)
    if (n(i) > 0)
        a(i) = 1;
    else
        a(i) = .001;
    end
end
end

