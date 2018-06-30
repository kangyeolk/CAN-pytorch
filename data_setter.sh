# Temp
if [ $# -eq 0 ]
then
  eval "rm -r ./data/Images/*"
else
  # echo $1 $2 'will be used'
  file_1=$(eval "ls ./data/dog/ | grep -E '$1'")
  file_2=$(eval "ls ./data/dog/ | grep -E '$2'") # Double quote is necessary
  eval "cp -R ./data/dog/$file_1 ./data/Images/"
  eval "cp -R ./data/dog/$file_2 ./data/Images/"
  # eval 'ls ./data/Images'
fi
