#!/usr/bin/env bash

for game in 'video_pinball' 'boxing' 'breakout' 'star_gunner' 'robotank' 'atlantis' 'crazy_climber' 'gopher' 'demon_attack' 'name_this_game' 'krull' 'assault' 'road_runner' 'kangaroo' 'jamesbond' 'tennis' 'pong' 'space_invaders' 'beam_rider' 'tutankham'
do
    sbatch submit.sh $game
done