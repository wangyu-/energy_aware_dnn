#! /bin/sh

mkdir -p /tmp/yw7/
echo running
while [ 1 ]
do
#sleep 0.001
./measure_power.sh > /tmp/yw7/power_result.tmp
#./measure_power.sh |tee /tmp/yw7/power_result.tmp
mv /tmp/yw7/power_result.tmp /tmp/yw7/power_result
done
