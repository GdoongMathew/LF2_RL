class Lf2_Char:
    def  __init__(self, u='p', d=';', l='l', r="'", att='t', jump='y', defend='u'):
        self.u = u              # Up
        self.d = d              # Down
        self.l = l              # Left
        self.r = r              # Right
        self.att = att           # Attact
        self.jum = jump         # Jump
        self.defend = defend    # Defend
    def up(self):
        return self.u
    def down(self):
        return self.d
    def left(self):
        return self.l
    def right(self):
        return self.r
    def attact(self):
        return self.att
    def jump(self):
        return self.jum
    def deffend(self):
        return self.defend
    def sp_attact1(self):
        raise NotImplementedError
    def sp_attact2(self):
        raise NotImplementedError
    def sp_attact3(self):
        raise NotImplementedError
    def sp_attact4(self):
        raise NotImplementedError
    def sp_attact5(self):
        raise NotImplementedError
    def sp_attact6(self):
        return NotImplementedError

class John(Lf2_Char):
    def sp_attact1(self, direction='right'):
        # Energy Blast
        return self.deffend() + getattr(self, direction)() + self.attact()
    def sp_attact2(self):
        # Heal others
        return self.deffend() + self.up() + self.jump()
    def sp_attact3(self, direction='right'):
        # Energy Shield
        return self.deffend() + getattr(self, direction)() + self.jump()
    def sp_attact4(self):
        # Energy Disk
        return self.deffend() + self.up() + self.attact()
    def sp_attact5(self):
        # Heal myself
        return self.deffend() + self.down() + self.jump()

class Firen(Lf2_Char):
    def sp_attact1(self, direction='right', num=1):
        # Fire Ball
        return self.deffend() + getattr(self, direction)() + num * self.attact()
    def sp_attact2(self, direction='right'):
        # Blaze
        return self.deffend() + getattr(self, direction)() + self.jump()
    def sp_attact3(self):
        # Inferno
        return self.deffend() + self.down() + self.jump()
    def sp_attact4(self):
        # Explosion
        return self.down() + self.up() + self.jump()

class Freeze(Lf2_Char):
    def sp_attact1(self, direction='right', num=1):
        # Ice Blast
        return self.deffend() + getattr(self, direction)() + num * self.attact()
    def sp_attact2(self):
        # Ice Sword
        return self.deffend() + self.down() + self.jump()
    def sp_attact3(self, direction='right'):
        # Icicle
        return self.deffend() + getattr(self, direction)() + self.jump()
    def sp_attact4(self):
        # Whirlwind
        return self.deffend() + self.up() + self.jump()

class Julian(Lf2_Char):
    def sp_attact1(self, direction='right'):
        # Soul Punch
        return getattr(self, direction)() + getattr(self, direction)() + self.attact()
    def sp_attact2(self):
        # Uppercut
        return self.deffend() + self.up() + self.attact()
    def sp_attact3(self, direction='right', num=1):
        # Skull Blast
        return self.deffend() + getattr(self, direction)() + self.attact() * num
    def sp_attact4(self, num=0):
        # Mirror Image
        return self.deffend() + self.jump() + self.attact() + self.jump() * num
    def sp_attact5(self):
        # Big Bang
        return self.deffend() + self.up() + self.jump()
    def sp_attact6(self, direction='right'):
        # Soul Bomb
        return self.deffend() + getattr(self, direction)() + self.jump()


if  __name__ == '__main__':
    p1 = Julian()
    print(p1.sp_attact5())