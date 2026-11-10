sed -i -e '$a\ ' README.md
git remote add origin https://github.com/kaist-dmlab/CluSTER.git
git add *
git commit -m "Update README.md"
git push  origin main
git config credential.helper store
git config credential.helper cache
git config credential.helper 'cache --timeout=180000'

