function out = center( data,f )


power = sum(data)*0.5;
add_power = 0;
for i=1:length(data)

    add_power = data(i)+ add_power;
    if add_power>power || add_power==power
        break
    end
end
out = f(i);


