f() {
    local x=$1
    local y=$2
    echo "$(bc -l <<< "$y - $x^2 + 1")"  # Ejemplo: y' = y - x^2 + 1
}


x0=0
y0=0.5
h=0.1 
n=10  



x=$x0
y=$y0

for ((i=0; i<=n; i++)); do
    printf "%d\t %.4f\t %.4f\n" $i $x $y  
    y_new=$(bc -l <<< "$y + $h * $(f $x $y)")
    x=$(bc -l <<< "$x + $h") 
    y=$y_new  
done

