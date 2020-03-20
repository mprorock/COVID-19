
git remote add upstream https://github.com/CSSEGISandData/COVID-19.git

git add .
git commit -a -m "updates"; git fetch upstream ; git merge upstream/master
