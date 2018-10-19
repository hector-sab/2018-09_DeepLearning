# Original Author: Shivam Shekhar
# https://github.com/shivamshekhar/Chrome-T-Rex-Rush
# Modification by: Hector Sanchez

import cv2
import random
import numpy as np
import pygame as pg

import utils as ut
import constants as ct


class Dino:
    def __init__(self,screen,sizex=-1,sizey=-1):
        self.screen = screen

        self.checkPoint_sound = pg.mixer.Sound('sprites/checkPoint.wav')
        self.images,self.rect = ut.load_sprite_sheet('dino.png',5,1,sizex,sizey,-1)
        self.images1,self.rect1 = ut.load_sprite_sheet('dino_ducking.png',2,1,59,sizey,-1)
        self.rect.bottom = int(0.98*ct.height)
        self.rect.left = ct.width/15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0,0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98*ct.height):
            self.rect.bottom = int(0.98*ct.height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + ct.gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1)%2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1)%2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2 + 2

        if self.isDead:
           self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index)%2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pg.mixer.get_init() != None:
                    self.checkPoint_sound.play()

        self.counter = (self.counter + 1)

class Cactus(pg.sprite.Sprite):
    def __init__(self,screen,speed=5,sizex=-1,sizey=-1):
        pg.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.images,self.rect = ut.load_sprite_sheet('cacti-small.png',3,1,sizex,sizey,-1)
        self.rect.bottom = int(0.98*ct.height)
        self.rect.left = ct.width + self.rect.width
        self.image = self.images[random.randrange(0,3)]
        self.movement = [-1*speed,0]

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()

class Ptera(pg.sprite.Sprite):
    def __init__(self,screen,speed=5,sizex=-1,sizey=-1):
        pg.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.images,self.rect = ut.load_sprite_sheet('ptera.png',2,1,sizex,sizey,-1)
        self.ptera_height = [ct.height*0.82,ct.height*0.75,ct.height*0.60]
        self.rect.centery = self.ptera_height[random.randrange(0,3)]
        self.rect.left = ct.width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1*speed,0]
        self.index = 0
        self.counter = 0

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index+1)%2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()

class Ground():
    def __init__(self,screen,speed=-5):
        self.screen = screen
        self.image,self.rect = ut.load_image('ground.png',-1,-1,-1)
        self.image1,self.rect1 = ut.load_image('ground.png',-1,-1,-1)
        self.rect.bottom = ct.height
        self.rect1.bottom = ct.height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        self.screen.blit(self.image,self.rect)
        self.screen.blit(self.image1,self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right

class Cloud(pg.sprite.Sprite):
    def __init__(self,screen,x,y):
        pg.sprite.Sprite.__init__(self,self.containers)
        self.screen = screen
        self.image,self.rect = ut.load_image('cloud.png',int(90*30/42),30,-1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1*self.speed,0]

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()

class Scoreboard():
    def __init__(self,screen,x=-1,y=-1):
        self.screen = screen
        self.score = 0
        self.tempimages,self.temprect = ut.load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        self.image = pg.Surface((55,int(11*6/5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = ct.width*0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = ct.height*0.1
        else:
            self.rect.top = y

    def draw(self):
        self.screen.blit(self.image,self.rect)

    def update(self,score):
        score_digits = ut.extractDigits(score)
        self.image.fill(ct.background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s],self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0



class TRexGame:
    def __init__(self):
        self.high_score = 0

    def set_controller(self,model,cam,im_shape):
        self.cam = cam
        self.model = model
        self.im_shape = im_shape

    def start(self):
        pg.init()
        self.screen = pg.display.set_mode(ct.scr_size)
        self.clock = pg.time.Clock()
    
        self.jump_sound = pg.mixer.Sound('sprites/jump.wav')
        self.die_sound = pg.mixer.Sound('sprites/die.wav')
        
        pg.display.set_caption('T-Rex Rush')
        
        self.cam.start()

        self.isGameQuit = self.introscreen()
        if not self.isGameQuit:
            self.gameplay()

    
    def disp_gameOver_msg(self,retbutton_image,gameover_image):
        retbutton_rect = retbutton_image.get_rect()
        retbutton_rect.centerx = ct.width / 2
        retbutton_rect.top = ct.height*0.52

        gameover_rect = gameover_image.get_rect()
        gameover_rect.centerx = ct.width / 2
        gameover_rect.centery = ct.height*0.35

        self.screen.blit(retbutton_image, retbutton_rect)
        self.screen.blit(gameover_image, gameover_rect)

    def introscreen(self):
        temp_dino = Dino(self.screen,47,47)
        temp_dino.isBlinking = True
        gameStart = False

        callout,callout_rect = ut.load_image('call_out.png',196,45,-1)
        callout_rect.left = ct.width*0.05
        callout_rect.top = ct.height*0.4

        temp_ground,temp_ground_rect = ut.load_sprite_sheet('ground.png',15,1,-1,-1,-1)
        temp_ground_rect.left = ct.width/20
        temp_ground_rect.bottom = ct.height

        logo,logo_rect = ut.load_image('logo.png',240,40,-1)
        logo_rect.centerx = ct.width*0.6
        logo_rect.centery = ct.height*0.6
        while not gameStart:
            if pg.display.get_surface() == None:
                print("Couldn't load display surface")
                return(True)
            else:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        return(True)
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_SPACE or event.key == pg.K_UP:
                            temp_dino.isJumping = True
                            temp_dino.isBlinking = False
                            temp_dino.movement[1] = -1*temp_dino.jumpSpeed

            temp_dino.update()

            if pg.display.get_surface() != None:
                self.screen.fill(ct.background_col)
                self.screen.blit(temp_ground[0],temp_ground_rect)
                if temp_dino.isBlinking:
                    self.screen.blit(logo,logo_rect)
                    self.screen.blit(callout,callout_rect)
                temp_dino.draw()

                pg.display.update()

            self.clock.tick(ct.FPS)
            if temp_dino.isJumping == False and temp_dino.isBlinking == False:
                gameStart = True

    def gameplay(self):
        gamespeed = 4
        startMenu = False
        gameOver = False
        gameQuit = False
        playerDino = Dino(self.screen,44,47)
        new_ground = Ground(self.screen,-1*gamespeed)
        scb = Scoreboard(self.screen)
        highsc = Scoreboard(self.screen,ct.width*0.78)
        counter = 0

        cacti = pg.sprite.Group()
        pteras = pg.sprite.Group()
        clouds = pg.sprite.Group()
        last_obstacle = pg.sprite.Group()

        Cactus.containers = cacti
        Ptera.containers = pteras
        Cloud.containers = clouds

        retbutton_image,retbutton_rect = ut.load_image('replay_button.png',35,31,-1)
        gameover_image,gameover_rect = ut.load_image('game_over.png',190,11,-1)

        temp_images,temp_rect = ut.load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        HI_image = pg.Surface((22,int(11*6/5)))
        HI_rect = HI_image.get_rect()
        HI_image.fill(ct.background_col)
        HI_image.blit(temp_images[10],temp_rect)
        temp_rect.left += temp_rect.width
        HI_image.blit(temp_images[11],temp_rect)
        HI_rect.top = ct.height*0.1
        HI_rect.left = ct.width*0.73

        while not gameQuit:
            while startMenu:
                pass
            while not gameOver:
                if pg.display.get_surface() == None:
                    print("Couldn't load display surface")
                    gameQuit = True
                    gameOver = True
                else:
                    ##########

                    frame = self.cam.get_frame()
                    frame = cv2.resize(frame,self.im_shape)
                    frame = np.reshape(frame,(1,self.im_shape[0],self.im_shape[1],3))
                    out = self.model.predict(frame)
                    CLS = {0:'Center',1:'Top',2:'Bottom'}
                    print(CLS[int(out[0])],)

                    if True:
                        for event in pg.event.get():
                            if event.type==pg.QUIT:
                                gameQuit = True
                                gameOver = True

                        if out[0]==1:
                            if playerDino.rect.bottom == int(0.98*ct.height):
                                playerDino.isJumping = True
                                if pg.mixer.get_init() != None:
                                    self.jump_sound.play()
                                playerDino.movement[1] = -1*playerDino.jumpSpeed
                        elif out[0]==0:
                            playerDino.isDucking = False
                        elif out[0]==2:
                            if not (playerDino.isJumping and playerDino.isDead):
                                playerDino.isDucking = True

                    if False:
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                gameQuit = True
                                gameOver = True

                            if event.type == pg.KEYDOWN:
                                if event.key == pg.K_SPACE:
                                    if playerDino.rect.bottom == int(0.98*ct.height):
                                        playerDino.isJumping = True
                                        if pg.mixer.get_init() != None:
                                            self.jump_sound.play()
                                        playerDino.movement[1] = -1*playerDino.jumpSpeed

                                if event.key == pg.K_DOWN:
                                    if not (playerDino.isJumping and playerDino.isDead):
                                        playerDino.isDucking = True

                            if event.type == pg.KEYUP:
                                if event.key == pg.K_DOWN:
                                    playerDino.isDucking = False
                    ##########


                for c in cacti:
                    c.movement[0] = -1*gamespeed
                    if pg.sprite.collide_mask(playerDino,c):
                        playerDino.isDead = True
                        if pg.mixer.get_init() != None:
                            self.die_sound.play()

                for p in pteras:
                    p.movement[0] = -1*gamespeed
                    if pg.sprite.collide_mask(playerDino,p):
                        playerDino.isDead = True
                        if pg.mixer.get_init() != None:
                            self.die_sound.play()

                if len(cacti) < 2:
                    if len(cacti) == 0:
                        last_obstacle.empty()
                        last_obstacle.add(Cactus(self.screen,gamespeed,40,40))
                    else:
                        for l in last_obstacle:
                            if l.rect.right < ct.width*0.7 and random.randrange(0,50) == 10:
                                last_obstacle.empty()
                                last_obstacle.add(Cactus(self.screen,gamespeed, 40, 40))

                if len(pteras) == 0 and random.randrange(0,200) == 10 and counter > 500:
                    for l in last_obstacle:
                        if l.rect.right < ct.width*0.8:
                            last_obstacle.empty()
                            last_obstacle.add(Ptera(self.screen,gamespeed, 46, 40))

                if len(clouds) < 5 and random.randrange(0,300) == 10:
                    Cloud(self.screen,ct.width,random.randrange(ct.height/5,ct.height/2))

                playerDino.update()
                cacti.update()
                pteras.update()
                clouds.update()
                new_ground.update()
                scb.update(playerDino.score)
                highsc.update(self.high_score)

                if pg.display.get_surface() != None:
                    self.screen.fill(ct.background_col)
                    new_ground.draw()
                    clouds.draw(self.screen)
                    scb.draw()
                    if self.high_score != 0:
                        highsc.draw()
                        self.screen.blit(HI_image,HI_rect)
                    cacti.draw(self.screen)
                    pteras.draw(self.screen)
                    playerDino.draw()

                    pg.display.update()
                self.clock.tick(ct.FPS)

                if playerDino.isDead:
                    gameOver = True
                    if playerDino.score > self.high_score:
                        self.high_score = playerDino.score

                if counter%700 == 699:
                    new_ground.speed -= 1
                    gamespeed += 1

                counter = (counter + 1)

            if gameQuit:
                break

            while gameOver:
                if pg.display.get_surface() == None:
                    print("Couldn't load display surface")
                    gameQuit = True
                    gameOver = False
                else:
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            gameQuit = True
                            gameOver = False
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_ESCAPE:
                                gameQuit = True
                                gameOver = False

                            if event.key == pg.K_RETURN or event.key == pg.K_SPACE:
                                gameOver = False
                                self.gameplay()
                highsc.update(self.high_score)
                if pg.display.get_surface() != None:
                    self.disp_gameOver_msg(retbutton_image,gameover_image)
                    if self.high_score != 0:
                        highsc.draw()
                        self.screen.blit(HI_image,HI_rect)
                    pg.display.update()
                self.clock.tick(ct.FPS)

        pg.quit()
        self.cam.stop()
        quit()
