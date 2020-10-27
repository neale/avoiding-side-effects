
cd append_spawn
mkdir images1
mkdir images2
mkdir images3
mkdir images4
mkdir images5
ffmpeg -i "aup.mp4" images1/out-%03d.png
ffmpeg -i "naive.mp4" images2/out-%03d.png
ffmpeg -i "aup-p.mp4" images3/out-%03d.png
ffmpeg -i "dqn.mp4" images4/out-%03d.png
ffmpeg -i "ppo.mp4" images5/out-%03d.png

convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images1"/*.png "aup.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images2"/*.png "naive.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images3"/*.png "aup-p.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images4"/*.png "dqn.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images5"/*.png "ppo.gif"

cd ../append_still-easy
mkdir images1
mkdir images2
mkdir images3
mkdir images4
mkdir images5
ffmpeg -i "aup.mp4" images1/out-%03d.png
ffmpeg -i "naive.mp4" images2/out-%03d.png
ffmpeg -i "aup-p.mp4" images3/out-%03d.png
ffmpeg -i "dqn.mp4" images4/out-%03d.png
ffmpeg -i "ppo.mp4" images5/out-%03d.png

convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images1"/*.png "aup.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images2"/*.png "naive.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images3"/*.png "aup-p.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images4"/*.png "dqn.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images5"/*.png "ppo.gif"

cd ../prune_still-easy
mkdir images1
mkdir images2
mkdir images3
mkdir images4
mkdir images5
ffmpeg -i "aup.mp4" images1/out-%03d.png
ffmpeg -i "naive.mp4" images2/out-%03d.png
ffmpeg -i "aup-p.mp4" images3/out-%03d.png
ffmpeg -i "dqn.mp4" images4/out-%03d.png
ffmpeg -i "ppo.mp4" images5/out-%03d.png

convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images1"/*.png "aup.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images2"/*.png "naive.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images3"/*.png "aup-p.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images4"/*.png "dqn.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images5"/*.png "ppo.gif"

cd ../append_still
mkdir images1
mkdir images2
mkdir images3
mkdir images4
mkdir images5
ffmpeg -i "aup.mp4" images1/out-%03d.png
ffmpeg -i "naive.mp4" images2/out-%03d.png
ffmpeg -i "aup-p.mp4" images3/out-%03d.png
ffmpeg -i "dqn.mp4" images4/out-%03d.png
ffmpeg -i "ppo.mp4" images5/out-%03d.png

convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images1"/*.png "aup.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images2"/*.png "naive.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images3"/*.png "aup-p.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images4"/*.png "dqn.gif"
convert -delay 5 -loop 0 -layers optimize -limit memory 64 "images5"/*.png "ppo.gif"
