#!/bin/bash
# find training_autumn_all/labels/ -name "chicago*" | xargs -i cp {} all_city/autumn/chicago/labels/
# find training_autumn_all/labels/ -name "paris*" | xargs -i cp {} all_city/autumn/paris/labels/
# find training_autumn_all/labels/ -name "zurich*" | xargs -i cp {} all_city/autumn/zurich/labels/

# find training_autumn_all/images/ -name "paris*" | xargs -i cp {} all_city/autumn/paris/images/
find training_autumn_all/images/ -name "zurich*" | xargs -i cp {} all_city/autumn/zurich/images/

find training_autumn_all/images/ -name "chicago*" | xargs -i cp {} all_city/autumn/chicago/images/
find training_spring/labels/ -name "berlin*" | xargs -i cp {} all_city/spring/berlin/labels/
find training_spring/labels/ -name "potsdam*" | xargs -i cp {} all_city/spring/potsdam/labels/

find training_spring/images/ -name "berlin*" | xargs -i cp {} all_city/spring/berlin/images/
find training_spring/images/ -name "potsdam*" | xargs -i cp {} all_city/spring/potsdam/images/