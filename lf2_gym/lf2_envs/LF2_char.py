# http://www.cheatbook.de/files/littlef2.htm

vk = {
    0x31: "1", 0x32: "2", 0x33: "3", 0x34: "4", 0x35: "5",
    0x36: "6", 0x37: "7", 0x38: "8", 0x39: "9", 0x30: "0",
    0x41: "A", 0x42: "B", 0x43: "C", 0x44: "D", 0x45: "E",
    0x46: "F", 0x47: "G", 0x48: "H", 0x49: "I", 0x4A: "J",
    0x4B: "K", 0x4C: "L", 0x4D: "M", 0x4E: "N", 0x4F: "O",
    0x50: "P", 0x51: "Q", 0x52: "R", 0x53: "S", 0x54: "T",
    0x55: "U", 0x56: "V", 0x57: "W", 0x58: "X", 0x59: "Y",
    0x5A: "Z" , 0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4",
    0x74: "F5", 0x75: "F6", 0x76: "F7", 0x77: "F8", 0x78: "F9",
    0x79: "F10", 0x7A: "F11", 0x7B: "F12", 0x26: "UP", 0x25: "LEFT",
    0x27: "RIGHT", 0x28: "DOWN", 0x1B: "ESC", 0x20: "SPACE",
    0x0D: "ENTER", 0x2D: "INSERT", 0x2E: "DELETE",  0x09: "TAB",
    0xA2: "CTRL", 0xA3: "CONTROL", 0xA0: "SHIFT", 0x14: "CAPSLOCK",
    0xBD: "subtract", 0xDB: "[", 0xDD: "]", 0xBA: ";", 0xDE: "'",
    0xC0: "`", 0xDC: "\\", 0xBC: ",", 0xBE: ".", 0xBF: "/"
    }


class Lf2Char:
    def __init__(self, player_id=1, ctrl_file=r'D:\Programs\Little_Fighter\data\control.txt'):

        if player_id > 3:
            raise ValueError("player_id can't be larger than 3.")

        with open(ctrl_file) as file:
            for i, line in enumerate(file):
                if i == player_id:
                    ctrl_code = [int(x) for x in line.split(' ') if x not in (' ', '\n')]
                    break

        self.u = vk[ctrl_code[1]]              # Up
        self.d = vk[ctrl_code[2]]              # Down
        self.l = vk[ctrl_code[3]]              # Left
        self.r = vk[ctrl_code[4]]              # Right
        self.att = vk[ctrl_code[5]]            # Attact
        self.jum = vk[ctrl_code[6]]            # Jump
        self.defend = vk[ctrl_code[7]]         # Defend

    def action_space(self):
        """
        Get the action space of a character.
        :return: return a list of possible actions.
        """
        # Remove all
        # Probably a better and more elegant way of doing this?
        action_list = [func for func in dir(self) if
                       (callable(getattr(self, func)) and
                        func != 'action_space' and
                        func[:2] != '__')]
        return action_list

    def idle(self):
        return None

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

    def run(self, direction='right'):
        return getattr(self, direction)() + getattr(self, direction)()


class John(Lf2Char):
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


class Firen(Lf2Char):
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


class Freeze(Lf2Char):
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


class Jan(Lf2Char):
    def sp_attact1(self):
        # Devil's Judgement
        return self.deffend() + self.up() + self.attact()

    def sp_attact2(self):
        # Angel's Blessing
        return self.deffend() + self.up() + self.jump()


class Justin(Lf2Char):
    def sp_attact1(self, num=1):
        # Wolf Punch
        return self.deffend() + self.down() + self.attact() * 1

    def sp_attact2(self, direction='right'):
        # Energy Blast
        return self.deffend() + getattr(self, direction)() + self.attact()


class Bat(Lf2Char):
    def sp_attact1(self, direction='right'):
        # Speed Punch
        return self.deffend() + getattr(self, direction)() + self.jump()

    def sp_attact2(self, direction='right'):
        # Eye Laser
        return self.deffend() + getattr(self, direction)() + self.attact()

    def sp_attact3(self):
        # Sommon Bats
        return self.deffend() + self.up() + self.jump()


class Julian(Lf2Char):
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


if __name__ == '__main__':
    p1 = Julian()
    print(p1.action_space())